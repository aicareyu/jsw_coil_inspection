from __future__ import annotations

from pathlib import Path

import cv2

from inspection.classes import DEFECT_CLASSES
from inspection.detector import Detector


def _first_sample(project_root: Path, folder: str) -> Path | None:
    directory = project_root / "samples" / folder
    files = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg")) + sorted(
        directory.glob("*.jpeg")
    )
    return files[0] if files else None


def test_three_defect_classes_have_display_names():
    assert DEFECT_CLASSES["foreign_edge"] == "FOREIGN EDGE"
    assert DEFECT_CLASSES["edge_rub"] == "EDGE RUB"
    assert DEFECT_CLASSES["loose_coil"] == "LOOSE COIL"


def test_edge_rub_sample_reports_edge_rub(project_root: Path):
    path = _first_sample(project_root, "edge_rub")
    if path is None:
        return
    result = Detector().detect(cv2.imread(str(path)))
    assert result["status"] == "FAIL"
    assert result["defects"][0]["display_name"] == "EDGE RUB"


def test_uploaded_edge_rub_sheet_is_not_foreign_edge(project_root: Path):
    path = project_root / "samples" / "edge_rub" / "edge_rub_13.jpeg"
    if not path.exists():
        return
    result = Detector().detect(cv2.imread(str(path)))
    assert result["status"] == "FAIL"
    names = {defect["display_name"] for defect in result["defects"]}
    assert names == {"EDGE RUB"}


def test_loose_coil_box_covers_coil_not_border(project_root: Path):
    path = Path("/home/careyu/Downloads/loose-coil.jpeg")
    sample = project_root / "samples" / "loose_coil" / "loose_coil_07.jpeg"
    image_path = path if path.exists() else sample
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    result = Detector().detect(image)
    assert result["status"] == "FAIL"
    assert result["defects"][0]["display_name"] == "LOOSE COIL"
    height, width = image.shape[:2]
    x1, y1, x2, y2 = result["defects"][0]["bbox"]
    area = (x2 - x1) * (y2 - y1)
    assert area > 0.22 * width * height
    assert (x2 - x1) > 0.40 * width
    assert (y2 - y1) > 0.40 * height
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    assert 0.30 * width < cx < 0.70 * width
    assert 0.25 * height < cy < 0.70 * height


def test_loose_coil_sample_reports_loose_coil(project_root: Path):
    path = _first_sample(project_root, "loose_coil")
    if path is None:
        return
    result = Detector().detect(cv2.imread(str(path)))
    assert result["status"] == "FAIL"
    assert result["defects"][0]["display_name"] == "LOOSE COIL"
