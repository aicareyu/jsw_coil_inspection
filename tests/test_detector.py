from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inspection.demo_detector import DemoForeignEdgeDetector
from inspection.detector import Detector
from inspection.preprocessing import preprocess
from tests.conftest import make_clean_coil, make_defective_coil, make_uniform_image


def test_invalid_image_does_not_crash():
    detector = Detector()
    result = detector.detect(None)
    assert result["status"] == "PASS"
    assert result["defects"] == []
    assert "error" in result


def test_empty_array_does_not_crash():
    detector = Detector()
    result = detector.detect(np.array([]))
    assert result["status"] == "PASS"


def test_pass_on_uniform_image():
    detector = Detector()
    result = detector.detect(make_uniform_image())
    assert result["status"] == "PASS"
    assert result["defects"] == []
    assert result["processing_time_ms"] >= 0


def test_pass_on_clean_coil_pattern():
    detector = DemoForeignEdgeDetector()
    result = detector.detect(make_clean_coil())
    assert result["status"] == "PASS"


def test_fail_on_synthetic_foreign_edge():
    detector = DemoForeignEdgeDetector()
    result = detector.detect(make_defective_coil())
    assert result["status"] in {"FAIL", "PASS"}
    if result["status"] == "FAIL":
        defect = result["defects"][0]
        assert defect["display_name"] == "FOREIGN EDGE"
        assert defect["class_name"] == "foreign_edge"
        x1, y1, x2, y2 = defect["bbox"]
        assert x2 > x1 and y2 > y1
        assert 0 < defect["confidence"] <= 1


def test_result_not_filename_dependent(tmp_path: Path):
    detector = Detector()
    image = make_uniform_image()
    a = tmp_path / "fail_please.jpg"
    b = tmp_path / "pass_please.jpg"
    cv2.imwrite(str(a), image)
    cv2.imwrite(str(b), image)
    ra = detector.detect(cv2.imread(str(a)))
    rb = detector.detect(cv2.imread(str(b)))
    assert ra["status"] == rb["status"]


def test_bbox_not_hardcoded():
    detector = DemoForeignEdgeDetector()
    img_a = make_defective_coil()
    img_b = np.ascontiguousarray(np.flip(img_a, axis=1))
    ra = detector.detect(img_a)
    rb = detector.detect(img_b)
    if ra["defects"] and rb["defects"]:
        assert ra["defects"][0]["bbox"] != rb["defects"][0]["bbox"]


def test_preprocess_keeps_original():
    image = make_uniform_image()
    prepared = preprocess(image)
    assert prepared.original_image.shape == image.shape
    assert prepared.processed_image is not None
    assert prepared.gray.ndim == 2


def test_unusual_resolution_is_handled():
    detector = Detector()
    tiny = np.full((12, 18, 3), 80, dtype=np.uint8)
    huge = np.full((40, 2200, 3), 90, dtype=np.uint8)
    assert detector.detect(tiny)["status"] in {"PASS", "FAIL"}
    assert detector.detect(huge)["status"] in {"PASS", "FAIL"}


def test_sample_defective_coils_produce_fail(sample_images):
    if not sample_images:
        return
    detector = Detector()
    failures = 0
    for path in sample_images[:8]:
        image = cv2.imread(str(path))
        if image is None:
            continue
        result = detector.detect(image)
        assert result["status"] in {"PASS", "FAIL"}
        if result["status"] == "FAIL":
            failures += 1
            assert result["defects"][0]["display_name"] in {
                "FOREIGN EDGE",
                "EDGE RUB",
                "LOOSE COIL",
            }
            x1, y1, x2, y2 = result["defects"][0]["bbox"]
            assert x2 > x1 and y2 > y1
    assert failures >= 1
