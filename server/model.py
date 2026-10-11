"""Prediction backend: a trained checkpoint (torch), or a random fake for UI work."""
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CATEGORY_MAP_PATH = MODEL_DIR / "category_map.json"

# Must match training (resize straight to 224x224, ImageNet normalization). Do not change.
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Reference order from docs/DATA.md. Only used by fake mode; the real model's own list wins.
FAKE_CLASSES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal",
    "paper", "plastic", "shoes", "trash_general", "trash_infectious", "white-glass",
]
FAKE_CATEGORY_MAP = {
    "battery": "hazardous", "biological": "organic", "brown-glass": "recyclable",
    "cardboard": "recyclable", "clothes": "general", "green-glass": "recyclable",
    "metal": "recyclable", "paper": "recyclable", "plastic": "recyclable",
    "shoes": "general", "trash_general": "general", "trash_infectious": "infectious",
    "white-glass": "recyclable",
}


def load_json(name):
    with open(ROOT / "data" / name, encoding="utf-8") as f:
        return json.load(f)


class ModelUnavailable(Exception):
    pass


@dataclass
class Prediction:
    top_class: str
    top_confidence: float
    runner_class: str
    runner_confidence: float


class FakePredictor:
    fake = True

    def __init__(self, threshold):
        self.classes = list(FAKE_CLASSES)
        self.category_map = dict(FAKE_CATEGORY_MAP)
        self.threshold = threshold

    def predict(self, image):
        top, runner = random.sample(self.classes, 2)
        if random.random() < 0.2 or self.threshold >= 0.95:  # about 1 in 5 results land in "uncertain"
            top_conf = random.uniform(0.30, min(0.94, max(0.31, self.threshold - 0.05)))
            runner_conf = random.uniform(0.2, top_conf) if top_conf > 0.2 else top_conf * 0.5
        else:
            top_conf = random.uniform(max(self.threshold + 0.05, 0.65), 0.98)
            runner_conf = random.uniform(0.0, min(1 - top_conf, top_conf * 0.5))
        runner_conf = min(runner_conf, 1 - top_conf)
        return Prediction(top, round(top_conf, 4), runner, round(runner_conf, 4))


class TorchPredictor:
    """Loads a checkpoint saved by the training team (see docs/DATA.md).

    Checkpoint keys: model_name, class_order, state_dict. The category table lives in
    models/category_map.json. Preprocessing must match training: direct resize, no crop.
    """
    fake = False

    def __init__(self, path, category_map_path=CATEGORY_MAP_PATH):
        try:
            import torch
            from torch import nn
            from torchvision import models, transforms
        except ImportError as e:
            raise ModelUnavailable("ยังไม่ได้ติดตั้ง torch/torchvision (pip install -r requirements.txt)") from e
        try:
            ckpt = torch.load(path, map_location="cpu")
            self.classes = list(ckpt["class_order"])
            net = _build_net(ckpt["model_name"], len(self.classes), models, nn)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
        except Exception as e:
            raise ModelUnavailable(f"โหลดไฟล์โมเดล {Path(path).name} ไม่สำเร็จ: {e}") from e
        try:
            with open(category_map_path, encoding="utf-8") as f:
                self.category_map = json.load(f)
        except Exception as e:
            raise ModelUnavailable("อ่าน models/category_map.json ไม่ได้") from e
        missing = set(self.classes) - set(self.category_map)
        if missing:
            raise ModelUnavailable(f"models/category_map.json ไม่มีคลาส: {sorted(missing)}")
        self.model_name = ckpt["model_name"]
        self._torch = torch
        self._net = net
        self._tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def predict(self, image):
        from PIL import ImageOps
        # Phone photos keep rotation in EXIF; without this the model sees them sideways.
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self._tf(image).unsqueeze(0)
        with self._torch.inference_mode():
            probs = self._torch.softmax(self._net(x).float(), dim=1)[0]
        (p1, p2), (i1, i2) = self._torch.topk(probs, 2)
        return Prediction(self.classes[int(i1)], float(p1), self.classes[int(i2)], float(p2))


def _build_net(name, num_classes, models, nn):
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"ไม่รู้จักโมเดล {name}")
    return m


def find_checkpoint():
    """MODEL_PATH env, else models/model.pt, else the first models/*.pt. None -> fake mode."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env)
    if (MODEL_DIR / "model.pt").exists():
        return MODEL_DIR / "model.pt"
    found = sorted(MODEL_DIR.glob("*.pt"))
    return found[0] if found else None


_cache = {}


def get_predictor(threshold):
    """Fake mode when FAKE_MODEL=1 or no checkpoint is found in models/."""
    path = None if os.environ.get("FAKE_MODEL") == "1" else find_checkpoint()
    key = str(path) if path else "fake"
    if key not in _cache:
        _cache[key] = TorchPredictor(path) if path else FakePredictor(threshold)
    _cache[key].threshold = threshold
    return _cache[key]
