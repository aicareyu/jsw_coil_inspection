from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def sample_images(project_root: Path) -> list[Path]:
    folder = project_root / "samples" / "foreign_edge"
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    return files


def make_uniform_image(color: int = 140, size=(400, 500, 3)) -> np.ndarray:
    return np.full(size, color, dtype=np.uint8)


def make_clean_coil(size=(480, 640, 3)) -> np.ndarray:
    image = np.zeros(size, dtype=np.uint8)
    height, width = size[:2]
    for x in range(0, width, 8):
        cv2.line(image, (x, 0), (x, height), (160, 160, 160), 3)
        cv2.line(image, (x + 3, 0), (x + 3, height), (90, 90, 90), 1)
    return image


def make_defective_coil() -> np.ndarray:
    image = make_clean_coil()
    cv2.ellipse(image, (220, 180), (38, 16), 12, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(image, (228, 186), (18, 8), 18, 0, 360, (210, 210, 210), -1)
    pts = np.array([[200, 168], [250, 175], [246, 198], [198, 190]], dtype=np.int32)
    cv2.fillConvexPoly(image, pts, (15, 15, 15))
    return image
