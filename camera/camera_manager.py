from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class CameraManager:
    """USB / laptop camera wrapper with safe fallbacks.

    Industrial camera backends can be added later without changing the UI.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.capture: cv2.VideoCapture | None = None
        self.last_error: str | None = None

    @property
    def is_open(self) -> bool:
        return self.capture is not None and bool(self.capture.isOpened())

    @property
    def status(self) -> str:
        return "CAMERA CONNECTED" if self.is_open else "CAMERA NOT AVAILABLE"

    def start(self, index: int | None = None) -> bool:
        if index is not None:
            self.camera_index = int(index)
        self.stop()
        try:
            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                capture.release()
                self.last_error = f"Camera index {self.camera_index} is not available."
                return False
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                self.last_error = "Camera opened but no frame could be read."
                return False
            self.capture = capture
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.capture = None
            return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_open:
            return False, None
        try:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.last_error = "Lost camera frame."
                return False, None
            return True, frame
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

    def stop(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.capture = None

    def info(self) -> dict[str, Any]:
        return {
            "index": self.camera_index,
            "status": self.status,
            "connected": self.is_open,
            "error": self.last_error,
        }
