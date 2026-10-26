from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inspection.classes import DEFECT_CLASSES
from inspection.config import project_root
from inspection.preprocessing import preprocess

FEATURE_NAMES = (
    "orient_entropy",
    "color_ratio",
    "dark_blob",
    "edge_density",
    "gap_score",
    "peakiness",
    "log_sharp",
    "area_ratio",
)


def _orient_entropy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx) + np.pi
    hist, _ = np.histogram(angle, bins=18, range=(0, 2 * np.pi), weights=energy)
    probs = hist / (hist.sum() + 1e-6)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def extract_features(image: np.ndarray, bbox: list[int] | None = None) -> np.ndarray:
    prepared = preprocess(image, max_side=640, clahe_clip=2.0, denoise_h=3.0)
    gray = prepared.enhanced
    hsv = cv2.cvtColor(prepared.processed_image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_ratio = float((sat > 35).mean())
    dark = (val < 40).astype(np.uint8) * 255
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    dark_blob = float(stats[1:, cv2.CC_STAT_AREA].max() / val.size) if count > 1 else 0.0
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges.mean() / 255.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    gap_score = float(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).mean() / 255.0)
    resid = cv2.GaussianBlur(gray, (0, 0), 1.2)
    resid = np.abs(gray.astype(np.float32) - resid.astype(np.float32))
    height, width = gray.shape
    if bbox is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        sx = width / max(image.shape[1], 1)
        sy = height / max(image.shape[0], 1)
        x1 = int(np.clip(bbox[0] * sx, 0, width - 1))
        y1 = int(np.clip(bbox[1] * sy, 0, height - 1))
        x2 = int(np.clip(bbox[2] * sx, x1 + 1, width))
        y2 = int(np.clip(bbox[3] * sy, y1 + 1, height))
    region = resid[y1:y2, x1:x2]
    peakiness = float(region.max() / (region.mean() + 1e-6)) if region.size else 0.0
    sharp = float(cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_32F).var()) if region.size else 0.0
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(width * height, 1))
    return np.array(
        [
            _orient_entropy(gray),
            color_ratio,
            dark_blob,
            edge_density,
            gap_score,
            peakiness,
            np.log1p(sharp),
            area_ratio,
        ],
        dtype=np.float32,
    )


class DemoFeatureClassifier:
    def __init__(self, path: Path | None = None):
        self.path = path or (project_root() / "models" / "demo_classifier.npz")
        self.classes = list(DEFECT_CLASSES.keys())
        self.mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.std = np.ones(len(FEATURE_NAMES), dtype=np.float32)
        self.vectors = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        self.labels: list[str] = []
        self.loaded = False
        self.load()

    def load(self) -> bool:
        if not self.path.exists():
            return False
        data = np.load(self.path, allow_pickle=True)
        if "vectors" not in data.files or "labels" not in data.files:
            return False
        self.mean = data["mean"]
        self.std = data["std"]
        self.vectors = data["vectors"]
        self.labels = [str(item) for item in data["labels"]]
        self.loaded = len(self.labels) > 0
        return self.loaded

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            mean=self.mean,
            std=self.std,
            vectors=self.vectors,
            labels=np.array(self.labels),
        )

    def fit(self, features: np.ndarray, labels: list[str]) -> dict[str, Any]:
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        self.vectors = ((features - self.mean) / self.std).astype(np.float32)
        self.labels = list(labels)
        self.loaded = True
        self.save()
        predicted = [self.predict_from_vector(row) for row in features]
        correct = sum(int(pred == lab) for pred, lab in zip(predicted, labels))
        return {"samples": len(labels), "accuracy": correct / max(len(labels), 1)}

    def predict_from_vector(self, features: np.ndarray) -> str:
        if not self.loaded or self.vectors.size == 0:
            return "foreign_edge"
        normed = (features - self.mean) / self.std
        distances = np.linalg.norm(self.vectors - normed, axis=1)
        return str(self.labels[int(np.argmin(distances))])

    def predict(self, image: np.ndarray, bbox: list[int] | None = None) -> str:
        return self.predict_from_vector(extract_features(image, bbox))
