"""Loads a checkpoint in the training team's format (random weights) through the real torch path.

Skipped when torch is not installed. Accuracy can only be checked with the real model file.
"""
import io
import json

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

from server import model as server_model  # noqa: E402
from server.model import FAKE_CLASSES, ModelUnavailable, TorchPredictor  # noqa: E402


def make_checkpoint(path, class_order, model_name="efficientnet_b0"):
    # efficientnet_b0 keeps the test file small (~16 MB); convnext_tiny uses the same loader path.
    net = server_model._build_net(model_name, len(class_order), torchvision.models, torch.nn)
    torch.save({"model_name": model_name, "class_order": class_order, "state_dict": net.state_dict()}, path)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    order = list(reversed(FAKE_CLASSES))  # deliberately not the reference order
    make_checkpoint(d / "m.pt", order)
    return d / "m.pt", order


def test_loads_and_predicts(ckpt):
    path, order = ckpt
    p = TorchPredictor(path, server_model.CATEGORY_MAP_PATH)
    assert p.classes == order
    img = Image.new("RGB", (300, 120), (200, 30, 30))
    r = p.predict(img)
    assert r.top_class in order and r.runner_class in order and r.top_class != r.runner_class
    assert 0 < r.runner_confidence <= r.top_confidence <= 1


def test_exif_rotated_photo_is_accepted(ckpt):
    path, _ = ckpt
    p = TorchPredictor(path, server_model.CATEGORY_MAP_PATH)
    img = Image.new("RGB", (64, 32))
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    p.predict(Image.open(io.BytesIO(buf.getvalue())))


def test_missing_class_in_category_map_is_rejected(ckpt, tmp_path):
    path, _ = ckpt
    bad = tmp_path / "cm.json"
    bad.write_text(json.dumps({"plastic": "recyclable"}), encoding="utf-8")
    with pytest.raises(ModelUnavailable):
        TorchPredictor(path, bad)


def test_unknown_model_name_is_rejected(tmp_path):
    make_checkpoint(tmp_path / "x.pt", FAKE_CLASSES)
    ck = torch.load(tmp_path / "x.pt")
    ck["model_name"] = "vgg99"
    torch.save(ck, tmp_path / "x.pt")
    with pytest.raises(ModelUnavailable):
        TorchPredictor(tmp_path / "x.pt", server_model.CATEGORY_MAP_PATH)


def test_api_uses_real_model_when_checkpoint_present(ckpt, monkeypatch, client):
    path, _ = ckpt
    monkeypatch.delenv("FAKE_MODEL", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(path))
    server_model._cache.clear()
    try:
        health = client.get("/api/health").json()
        assert health["fake_model"] is False and health["classes"] == 13
        buf = io.BytesIO()
        Image.new("RGB", (50, 50), (10, 200, 10)).save(buf, "PNG")
        body = client.post("/api/predict", files={"file": ("a.png", buf.getvalue(), "image/png")}).json()
        assert body["fake_model"] is False
        cmap = json.loads(server_model.CATEGORY_MAP_PATH.read_text(encoding="utf-8"))
        assert body["category"] == cmap[body["class"]]
    finally:
        server_model._cache.clear()
