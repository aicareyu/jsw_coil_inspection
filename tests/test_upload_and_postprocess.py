from __future__ import annotations

import cv2
import numpy as np

from inspection.postprocessing import annotate_image, clamp_bbox
from tests.conftest import make_uniform_image


def test_annotate_draws_visible_box():
    image = make_uniform_image()
    result = {
        "status": "FAIL",
        "defects": [
            {
                "display_name": "FOREIGN EDGE",
                "confidence": 0.94,
                "bbox": [40, 50, 180, 140],
            }
        ],
    }
    annotated = annotate_image(image, result)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_clamp_bbox_stays_inside_image():
    box = clamp_bbox([-20, -5, 9000, 12], 200, 100)
    assert box[0] >= 0 and box[1] >= 0
    assert box[2] <= 200 and box[3] <= 100
    assert box[2] > box[0] and box[3] > box[1]


def test_pass_annotation_stamps_inspection_hud():
    image = make_uniform_image()
    annotated = annotate_image(image, {"status": "PASS", "defects": []})
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_fail_annotation_includes_inspection_hud():
    image = make_uniform_image()
    result = {
        "status": "FAIL",
        "inspection_id": "INS-TEST",
        "timestamp": "2026-09-09 12:00:00",
        "processing_time_ms": 120,
        "defects": [
            {
                "class_name": "loose_coil",
                "display_name": "LOOSE COIL",
                "bbox": [40, 50, 180, 140],
            }
        ],
    }
    annotated = annotate_image(image, result)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_supported_image_extensions():
    from ui.dashboard import IMAGE_TYPES, VIDEO_TYPES

    assert {"jpg", "jpeg", "png"} <= IMAGE_TYPES
    assert {"mp4", "avi"} <= VIDEO_TYPES
