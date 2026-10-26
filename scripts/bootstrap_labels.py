from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from inspection.classes import class_id
from inspection.config import project_root
from inspection.detector import Detector
from scripts.extract_samples import extract_all_samples


def _yolo_line(class_name: str, bbox: list[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = (x2 - x1) / max(width, 1)
    bh = (y2 - y1) / max(height, 1)
    return f"{class_id(class_name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def bootstrap_labels() -> int:
    root = project_root()
    extract_all_samples()
    detector = Detector()
    written = 0
    samples = []
    for folder in ("foreign_edge", "edge_rub", "loose_coil"):
        folder_path = root / "samples" / folder
        samples.extend(sorted(folder_path.glob("*.png")))
        samples.extend(sorted(folder_path.glob("*.jpg")))
        samples.extend(sorted(folder_path.glob("*.jpeg")))

    if not samples:
        print("No sample images found.")
        return 0

    val_every = 5
    for index, path in enumerate(samples, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        result = detector.detect(image)
        if not result.get("defects"):
            continue
        split = "val" if index % val_every == 0 else "train"
        image_dir = root / "dataset" / "images" / split
        label_dir = root / "dataset" / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        dest = image_dir / path.name
        shutil.copy2(path, dest)
        height, width = image.shape[:2]
        lines = [
            _yolo_line(defect["class_name"], defect["bbox"], width, height)
            for defect in result["defects"]
        ]
        (label_dir / f"{path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
        print(f"{split}: {path.name} -> {[d['display_name'] for d in result['defects']]}")
    print(f"Wrote {written} labeled images. Review them before training.")
    return written


if __name__ == "__main__":
    bootstrap_labels()
