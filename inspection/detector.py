from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inspection.config import load_config, project_root, resolve_path
from inspection.demo_detector import DemoForeignEdgeDetector
from inspection.postprocessing import annotate_image, attach_trigger_metadata, empty_result
from inspection.yolo_detector import YOLODetector

_DETECTOR: "Detector | None" = None


class Detector:
    """Replaceable inspection facade.

    Uses models/best.pt when present and loadable. Otherwise falls back to
    the classical Foreign Edge demo detector without crashing.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.model_path = resolve_path(self.config.get("paths", {}).get("model_file", "models/best.pt"))
        self.backend = "demo_cv"
        self.backend_label = "Demo CV"
        self.model_status = "DEMO MODE"
        self.status_detail = "DEMO MODE — TRAINED MODEL NOT INSTALLED"
        self.impl: DemoForeignEdgeDetector | YOLODetector = DemoForeignEdgeDetector(self.config)
        self._try_load_yolo()

    def _try_load_yolo(self) -> None:
        if not self.model_path.exists():
            return
        try:
            self.impl = YOLODetector(self.model_path, self.config)
            self.backend = "yolo"
            self.backend_label = "YOLO"
            self.model_status = "LOADED"
            self.status_detail = f"YOLO model loaded from {self.model_path.name}"
        except Exception as exc:
            self.impl = DemoForeignEdgeDetector(self.config)
            self.backend = "demo_cv"
            self.backend_label = "Demo CV"
            self.model_status = "DEMO MODE"
            self.status_detail = f"DEMO MODE — YOLO load failed ({exc})"

    def model_info(self) -> dict[str, Any]:
        exists = self.model_path.exists()
        return {
            "name": "Coil Defect Detector",
            "backend": self.backend_label,
            "backend_id": self.backend,
            "model_file": str(self.model_path.relative_to(project_root()))
            if _is_relative_to(self.model_path, project_root())
            else str(self.model_path),
            "model_exists": exists,
            "status": self.model_status,
            "detail": self.status_detail,
        }

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            result = empty_result(0.0)
            result["backend"] = self.backend
            result["error"] = "Invalid or empty image."
            return result
        try:
            result = self.impl.detect(image)
        except Exception as exc:
            result = empty_result(0.0)
            result["error"] = f"Inspection failed: {exc}"
        result.setdefault("backend", self.backend)
        result.setdefault("status", "PASS")
        result.setdefault("defects", [])
        result.setdefault("processing_time_ms", 0.0)
        for defect in result["defects"]:
            defect.setdefault("class_name", "foreign_edge")
            defect.setdefault("display_name", "FOREIGN EDGE")
        if result["defects"]:
            result["status"] = "FAIL"
        return attach_trigger_metadata(result)

    def inspect(self, image: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        result = self.detect(image)
        try:
            annotated = annotate_image(image, result)
        except Exception:
            annotated = image.copy() if isinstance(image, np.ndarray) else image
        return result, annotated


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def get_detector(force_reload: bool = False) -> Detector:
    global _DETECTOR
    clf_path = project_root() / "models" / "demo_classifier.npz"
    mtime = clf_path.stat().st_mtime if clf_path.exists() else 0
    stale = _DETECTOR is not None and getattr(_DETECTOR, "_clf_mtime", None) != mtime
    if _DETECTOR is None or force_reload or stale:
        _DETECTOR = Detector()
        _DETECTOR._clf_mtime = mtime
    return _DETECTOR
