from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inspection.classes import SAMPLE_FOLDERS
from inspection.config import project_root
from inspection.feature_classifier import DemoFeatureClassifier, extract_features
from scripts.extract_samples import extract_all_samples


def _collect() -> tuple[np.ndarray, list[str]]:
    root = project_root()
    extract_all_samples()
    features = []
    labels = []
    for class_name, folder in SAMPLE_FOLDERS.items():
        for path in sorted((root / "samples" / folder).iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            features.append(extract_features(image))
            labels.append(class_name)
    if not features:
        raise RuntimeError("No sample images available to train the demo classifier.")
    return np.stack(features), labels


def train_demo_classifier() -> dict:
    features, labels = _collect()
    classifier = DemoFeatureClassifier()
    stats = classifier.fit(features, labels)
    print(
        f"Trained demo classifier on {stats['samples']} samples "
        f"(train accuracy {stats['accuracy']*100:.0f}%)."
    )
    print(f"Saved {classifier.path}")
    return stats


if __name__ == "__main__":
    train_demo_classifier()
