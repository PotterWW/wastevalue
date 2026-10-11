import io
import json
from pathlib import Path

import pytest
from PIL import Image

from server.model import FAKE_CATEGORY_MAP, FAKE_CLASSES

ROOT = Path(__file__).resolve().parent.parent
FIELDS = {
    "class", "label_th", "confidence", "uncertain", "runner_up", "category", "category_th", "bin",
    "material", "sellable", "price_min", "price_max", "price_unit", "price_source", "price_date",
    "prep_steps", "disposal", "warning", "fake_model",
}


def image_bytes(fmt="PNG", size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 180, 90)).save(buf, fmt)
    return buf.getvalue()


def post(client, data, name="x.png", mime="image/png"):
    return client.post("/api/predict", files={"file": (name, data, mime)})


def test_waste_info_has_all_13_classes():
    info = json.loads((ROOT / "data" / "waste_info.json").read_text(encoding="utf-8"))
    assert set(FAKE_CLASSES) <= set(info)
    assert len([k for k in info if not k.startswith("_")]) == 13


def test_categories_cover_category_map():
    cats = json.loads((ROOT / "data" / "categories.json").read_text(encoding="utf-8"))
    assert set(FAKE_CATEGORY_MAP.values()) <= set(cats)


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"ok": True, "fake_model": True, "classes": 13, "threshold": 0.7}


def test_predict_returns_every_field(client):
    r = post(client, image_bytes())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == FIELDS
    assert body["class"] in FAKE_CLASSES
    assert body["fake_model"] is True
    assert set(body["runner_up"]) == {"class", "label_th", "confidence"}
    assert body["runner_up"]["class"] != body["class"]
    assert 0 <= body["confidence"] <= 1
    assert body["price_unit"] == "บาท/กก."


def test_predict_jpeg_accepted(client):
    assert post(client, image_bytes("JPEG"), "x.jpg", "image/jpeg").status_code == 200


def test_predict_consistent_with_tables(client):
    info = json.loads((ROOT / "data" / "waste_info.json").read_text(encoding="utf-8"))
    cats = json.loads((ROOT / "data" / "categories.json").read_text(encoding="utf-8"))
    for _ in range(60):
        b = post(client, image_bytes()).json()
        assert b["label_th"] == info[b["class"]]["label_th"]
        assert b["category"] == FAKE_CATEGORY_MAP[b["class"]]
        assert b["bin"] == cats[b["category"]]["bin"]
        assert b["warning"] == cats[b["category"]]["warning"]
        assert b["sellable"] == info[b["class"]]["sellable"]
        assert b["uncertain"] == (b["confidence"] < 0.7)
        if not b["sellable"]:
            assert b["price_min"] is None and b["price_max"] is None


def test_fake_mode_produces_both_certain_and_uncertain(client):
    flags = {post(client, image_bytes()).json()["uncertain"] for _ in range(100)}
    assert flags == {True, False}


def test_threshold_from_env(client, monkeypatch):
    monkeypatch.setenv("UNCERTAIN_THRESHOLD", "1.0")
    assert client.get("/api/health").json()["threshold"] == 1.0
    assert post(client, image_bytes()).json()["uncertain"] is True


def test_missing_file_is_400(client):
    r = client.post("/api/predict")
    assert r.status_code == 400 and r.json()["error"]
    r = client.post("/api/predict", data={"x": "y"})
    assert r.status_code == 400 and r.json()["error"]


@pytest.mark.parametrize("payload", [b"not an image", b"", b"\x89PNG\r\n\x1a\n broken"])
def test_non_image_is_400(client, payload):
    r = post(client, payload)
    assert r.status_code == 400 and r.json()["error"]


def test_unsupported_image_format_is_400(client):
    r = post(client, image_bytes("GIF"), "x.gif", "image/gif")
    assert r.status_code == 400 and "JPG" in r.json()["error"]


def test_too_large_is_413(client):
    r = post(client, b"\0" * (10 * 1024 * 1024 + 1))
    assert r.status_code == 413 and r.json()["error"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_favicon_does_not_404(client):
    assert client.get("/favicon.ico").status_code == 204


def test_prices_have_sources_and_only_for_sellable():
    info = json.loads((ROOT / "data" / "waste_info.json").read_text(encoding="utf-8"))
    for name, row in info.items():
        if name.startswith("_"):
            continue
        has_price = row["price_min"] is not None or row["price_max"] is not None
        if not row["sellable"]:
            assert not has_price and row["price_source"] is None, name
        if has_price:
            assert row["price_min"] is not None and row["price_max"] is not None, name
            assert 0 < row["price_min"] <= row["price_max"], name
            assert row["price_source"], f"{name}: price without a source"
            if row["price_date"] is not None:
                assert len(row["price_date"]) == 10, name  # YYYY-MM-DD


def test_demo_plastic_price_matches_data():
    info = json.loads((ROOT / "data" / "waste_info.json").read_text(encoding="utf-8"))["plastic"]
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert f'price_min: {info["price_min"]}, price_max: {info["price_max"]}' in html
    assert info["price_source"] in html
    assert f'price_date: "{info["price_date"]}"' in html
