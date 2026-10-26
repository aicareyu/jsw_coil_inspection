from __future__ import annotations

from pathlib import Path

import pytest

from inspection.detector import Detector
from inspection.yolo_detector import YOLODetector
from tests.conftest import make_uniform_image


def test_missing_model_uses_demo_backend(project_root: Path):
    assert not (project_root / "models" / "best.pt").exists()
    detector = Detector()
    info = detector.model_info()
    assert info["backend_id"] == "demo_cv"
    assert info["status"] == "DEMO MODE"
    assert "DEMO MODE" in info["detail"]


def test_yolo_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "best.pt"
    with pytest.raises(FileNotFoundError):
        YOLODetector(missing)


def test_detector_survives_without_cuda():
    detector = Detector()
    result = detector.detect(make_uniform_image())
    assert result["status"] == "PASS"
    assert result.get("backend") in {"demo_cv", "yolo"}
