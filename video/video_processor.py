from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from inspection.postprocessing import annotate_image


@dataclass
class VideoInspectionSummary:
    total_frames_inspected: int = 0
    defective_frames: int = 0
    foreign_edge_detections: int = 0
    detection_percentage: float = 0.0
    overall_result: str = "PASS"
    frames: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class VideoProcessor:
    def __init__(self, detector, frame_interval: int = 8, max_frames: int = 180):
        self.detector = detector
        self.frame_interval = max(1, int(frame_interval))
        self.max_frames = max(1, int(max_frames))

    def inspect(
        self,
        source: str | Path,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> VideoInspectionSummary:
        summary = VideoInspectionSummary()
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            summary.error = "Video cannot be opened."
            summary.overall_result = "PASS"
            return summary

        frame_index = 0
        inspected = 0
        try:
            while inspected < self.max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame_index % self.frame_interval != 0:
                    frame_index += 1
                    continue
                result = self.detector.detect(frame)
                annotated = annotate_image(frame, result)
                defect_count = len(result.get("defects", []))
                record = {
                    "frame_number": frame_index,
                    "status": result.get("status", "PASS"),
                    "defects": result.get("defects", []),
                    "defect_count": defect_count,
                    "processing_time_ms": result.get("processing_time_ms", 0.0),
                    "annotated": annotated,
                    "original": frame,
                }
                summary.frames.append(record)
                inspected += 1
                if result.get("status") == "FAIL":
                    summary.defective_frames += 1
                    summary.foreign_edge_detections += defect_count
                if progress_cb is not None:
                    progress_cb(record)
                frame_index += 1
        except Exception as exc:
            summary.error = f"Video inspection stopped: {exc}"
        finally:
            capture.release()

        summary.total_frames_inspected = inspected
        if inspected:
            summary.detection_percentage = 100.0 * summary.defective_frames / inspected
        summary.overall_result = "FAIL" if summary.defective_frames else "PASS"
        return summary


def write_tiny_test_video(path: Path, frames: list[np.ndarray], fps: int = 8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create a test video writer.")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path
