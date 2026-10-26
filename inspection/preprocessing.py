from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    original_image: np.ndarray
    processed_image: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    scale: float
    extras: dict[str, Any] = field(default_factory=dict)


def resize_keep_aspect(image: np.ndarray, max_side: int = 960) -> tuple[np.ndarray, float]:
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    min_v, max_v = float(image.min()), float(image.max())
    if max_v <= min_v:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_v) * (255.0 / (max_v - min_v))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, strength: float = 4.0) -> np.ndarray:
    h = max(1.0, float(strength))
    return cv2.fastNlMeansDenoising(gray, None, h, 7, 21)


def detect_edges(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blurred, 40, 120)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )


def red_annotation_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_a = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    lower_b = cv2.inRange(hsv, np.array([168, 70, 70]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(lower_a, lower_b)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)


def remove_red_annotations(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = red_annotation_mask(bgr)
    if float(mask.mean()) < 0.2:
        return bgr.copy(), mask
    cleaned = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return cleaned, mask


def preprocess(
    image: np.ndarray,
    max_side: int = 960,
    clahe_clip: float = 2.0,
    denoise_h: float = 4.0,
) -> PreprocessResult:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image: empty or unreadable.")
    if image.ndim == 2:
        original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        original = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        original = image.copy()

    resized, scale = resize_keep_aspect(original, max_side=max_side)
    cleaned, red_mask = remove_red_annotations(resized)
    gray = to_grayscale(cleaned)
    gray = denoise(gray, strength=denoise_h)
    enhanced = enhance_contrast(gray, clip_limit=clahe_clip)
    edges = detect_edges(enhanced)
    binary = adaptive_threshold(enhanced)

    return PreprocessResult(
        original_image=original,
        processed_image=cleaned,
        gray=gray,
        enhanced=enhanced,
        scale=scale,
        extras={
            "red_mask": red_mask,
            "edges": edges,
            "binary": binary,
            "resized_bgr": resized,
        },
    )
