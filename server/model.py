"""Prediction backend: a real MobileNetV2 checkpoint, or a random fake for UI work."""
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pt"

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
    fake = False

    def __init__(self, path):
        try:
            import torch
            from torch import nn
            from torchvision import models, transforms
        except ImportError as e:
            raise ModelUnavailable("ยังไม่ได้ติดตั้ง torch/torchvision (pip install -r requirements.txt)") from e
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.classes = list(ckpt["classes"])
            self.category_map = dict(ckpt["category_map"])
            size = int(ckpt.get("img_size", 224))
            net = models.mobilenet_v2(weights=None)
            net.classifier[1] = nn.Linear(1280, len(self.classes))
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
        except Exception as e:
            raise ModelUnavailable("โหลดไฟล์โมเดลไม่สำเร็จ ตรวจ models/model.pt") from e
        self._torch = torch
        self._net = net
        # Same preprocessing as training: direct resize (no crop), then normalize.
        self._tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(ckpt.get("mean", [0.485, 0.456, 0.406]), ckpt.get("std", [0.229, 0.224, 0.225])),
        ])

    def predict(self, image):
        x = self._tf(image.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            probs = self._torch.softmax(self._net(x), dim=1)[0]
        (p1, p2), (i1, i2) = self._torch.topk(probs, 2)
        return Prediction(self.classes[int(i1)], float(p1), self.classes[int(i2)], float(p2))


_cache = {}


def get_predictor(threshold):
    """Fake mode when FAKE_MODEL=1 or models/model.pt is missing."""
    use_fake = os.environ.get("FAKE_MODEL") == "1" or not MODEL_PATH.exists()
    key = "fake" if use_fake else "real"
    if key not in _cache:
        _cache[key] = FakePredictor(threshold) if use_fake else TorchPredictor(MODEL_PATH)
    _cache[key].threshold = threshold
    return _cache[key]
