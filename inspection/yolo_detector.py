from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from inspection.classes import display_name, normalize_class_name
from inspection.postprocessing import clamp_bbox


class YOLODetector:
    """Ultralytics YOLO inference wrapper. Loaded once and reused."""

    def __init__(self, model_path: str | Path, config: dict[str, Any] | None = None):
        infer = (config or {}).get("inference", {})
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self.conf_threshold = float(infer.get("conf_threshold", 0.45))
        self.iou_threshold = float(infer.get("iou_threshold", 0.45))
        self.max_side = int(infer.get("max_side", 960))
        self.device = self._resolve_device(str(infer.get("device", "auto")))
        self._model = self._load_model()

    def _resolve_device(self, requested: str) -> str:
        if requested and requested != "auto":
            return requested
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics is not available for YOLO inference.") from exc
        return YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "status": "PASS",
                "defects": [],
                "processing_time_ms": 0.0,
                "backend": "yolo",
            }
        height, width = image.shape[:2]
        defects: list[dict[str, Any]] = []
        try:
            results = self._model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.max_side,
                device=self.device,
                verbose=False,
            )
        except Exception:
            try:
                results = self._model.predict(
                    source=image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.max_side,
                    device="cpu",
                    verbose=False,
                )
                self.device = "cpu"
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return {
                    "status": "PASS",
                    "defects": [],
                    "processing_time_ms": round(elapsed_ms, 1),
                    "backend": "yolo",
                    "error": "YOLO inference failed; no detection returned.",
                }

        if results:
            result = results[0]
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
                clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
                for bbox, conf, cls_id in zip(xyxy, confs, clss):
                    raw_name = names.get(int(cls_id), "foreign_edge") if isinstance(names, dict) else "foreign_edge"
                    class_name = normalize_class_name(str(raw_name))
                    defects.append(
                        {
                            "class_name": class_name,
                            "display_name": display_name(class_name),
                            "confidence": round(float(conf), 3),
                            "bbox": clamp_bbox(bbox.tolist(), width, height),
                        }
                    )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "status": "FAIL" if defects else "PASS",
            "defects": defects,
            "processing_time_ms": round(elapsed_ms, 1),
            "backend": "yolo",
        }
