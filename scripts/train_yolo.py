from __future__ import annotations

import shutil
from pathlib import Path

from inspection.config import project_root


def _count_labels(root: Path) -> tuple[int, int]:
    train = list((root / "dataset" / "labels" / "train").glob("*.txt"))
    val = list((root / "dataset" / "labels" / "val").glob("*.txt"))
    return len(train), len(val)


def train() -> Path | None:
    root = project_root()
    data_yaml = root / "data.yaml"
    train_n, val_n = _count_labels(root)
    if train_n < 3:
        print("Not enough labeled training images.")
        print("Use the ANNOTATE tab or run: python -m scripts.bootstrap_labels")
        return None
    if val_n == 0:
        print("No validation labels found. Add at least one dataset/labels/val/*.txt file.")
        return None

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"Ultralytics is not available: {exc}")
        return None

    print(f"Training YOLO11 on {train_n} train / {val_n} val labels.")
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        project=str(root / "runs"),
        name="coil_defects",
        exist_ok=True,
    )
    best = Path(getattr(results, "save_dir", root / "runs" / "coil_defects")) / "weights" / "best.pt"
    if not best.exists():
        print("Training finished but best.pt was not found.")
        return None
    dest = root / "models" / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    print(f"Copied trained weights to {dest}")
    return dest


if __name__ == "__main__":
    train()
