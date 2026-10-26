from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from camera.camera_manager import CameraManager
from database.history import InspectionHistory
from inspection.detector import Detector
from tests.conftest import make_clean_coil, make_defective_coil, make_uniform_image
from video.video_processor import VideoProcessor, write_tiny_test_video


def test_camera_unavailable_does_not_crash():
    camera = CameraManager(camera_index=9999)
    started = camera.start()
    assert started is False
    assert camera.status == "CAMERA NOT AVAILABLE"
    ok, frame = camera.read()
    assert ok is False
    assert frame is None
    camera.stop()


def test_inspection_history_roundtrip(tmp_path: Path):
    history = InspectionHistory(tmp_path / "history.db", max_rows=10)
    history.add("coil_001.jpg", "FAIL", "FOREIGN EDGE", 0.94, 128.0, 1)
    history.add("coil_002.jpg", "PASS", None, None, 40.0, 0)
    rows = history.list_recent()
    assert len(rows) == 2
    assert rows[0]["image_name"] == "coil_002.jpg"
    assert rows[1]["defect_name"] == "FOREIGN EDGE"
    history.clear()
    assert history.list_recent() == []


def test_history_from_detector_result(tmp_path: Path):
    history = InspectionHistory(tmp_path / "history.db")
    detector = Detector()
    result = detector.detect(make_uniform_image())
    history.add_from_result("plain.png", result)
    row = history.list_recent(1)[0]
    assert row["result"] == "PASS"


def test_video_processing(tmp_path: Path):
    frames = [make_uniform_image(size=(240, 320, 3)) for _ in range(6)]
    frames[2] = cv2.resize(make_defective_coil(), (320, 240))
    video_path = tmp_path / "clip.mp4"
    try:
        write_tiny_test_video(video_path, frames, fps=6)
    except RuntimeError:
        video_path = tmp_path / "clip.avi"
        write_tiny_test_video(video_path, frames, fps=6)
    processor = VideoProcessor(Detector(), frame_interval=1, max_frames=10)
    summary = processor.inspect(video_path)
    assert summary.error is None
    assert summary.total_frames_inspected >= 1
    assert summary.overall_result in {"PASS", "FAIL"}
    assert 0.0 <= summary.detection_percentage <= 100.0


def test_invalid_video_path():
    processor = VideoProcessor(Detector(), frame_interval=1, max_frames=5)
    summary = processor.inspect("/tmp/does_not_exist_jsw_video.mp4")
    assert summary.error
    assert summary.total_frames_inspected == 0


def test_image_upload_decode_png(tmp_path: Path):
    path = tmp_path / "upload.png"
    cv2.imwrite(str(path), make_clean_coil())
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    result = Detector().detect(image)
    assert result["status"] in {"PASS", "FAIL"}
