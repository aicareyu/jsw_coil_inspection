from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import cv2
import numpy as np


DISPLAY_NAME = "FOREIGN EDGE"
TEXT_COLOR = (255, 255, 255)
BOX_RED = ((36, 36, 230), (20, 20, 190), (40, 40, 230))
CLASS_COLORS = {
    "foreign_edge": BOX_RED,
    "edge_rub": BOX_RED,
    "loose_coil": BOX_RED,
}


def clamp_bbox(bbox: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    ix1 = int(max(0, min(width - 1, round(x1))))
    iy1 = int(max(0, min(height - 1, round(y1))))
    ix2 = int(max(0, min(width, round(x2))))
    iy2 = int(max(0, min(height, round(y2))))
    if ix2 <= ix1:
        ix2 = min(width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(height, iy1 + 1)
    return [ix1, iy1, ix2, iy2]


def scale_bbox(
    bbox: Iterable[float],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> list[int]:
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    x1, y1, x2, y2 = bbox
    return clamp_bbox([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dst_w, dst_h)


def scale_result_to_original(
    result: dict[str, Any],
    processed_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> dict[str, Any]:
    scaled = dict(result)
    defects = []
    for defect in result.get("defects", []):
        item = dict(defect)
        item["bbox"] = scale_bbox(defect["bbox"], processed_shape, original_shape)
        defects.append(item)
    scaled["defects"] = defects
    return scaled


def _draw_label(
    image: np.ndarray,
    x1: int,
    y1: int,
    lines: list[str],
    box_color: tuple[int, int, int],
    label_bg: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    text_w = max(size[0] for size in sizes) if sizes else 80
    line_h = max(size[1] for size in sizes) + 8 if sizes else 18
    pad_x, pad_y = 8, 6
    box_w = text_w + pad_x * 2
    box_h = line_h * len(lines) + pad_y
    label_y1 = max(0, y1 - box_h - 4)
    label_y2 = label_y1 + box_h
    label_x2 = min(image.shape[1], x1 + box_w)
    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), label_bg, -1)
    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), box_color, 1)
    for i, line in enumerate(lines):
        ty = label_y1 + pad_y + (i + 1) * line_h - 8
        cv2.putText(image, line, (x1 + pad_x, ty), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def annotate_image(
    original_image: np.ndarray,
    result: dict[str, Any],
    highlight: bool = True,
) -> np.ndarray:
    annotated = original_image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    height, width = annotated.shape[:2]
    overlay = annotated.copy()

    for defect in result.get("defects", []):
        box_color, label_bg, overlay_color = CLASS_COLORS.get(
            str(defect.get("class_name", "foreign_edge")),
            CLASS_COLORS["foreign_edge"],
        )
        x1, y1, x2, y2 = clamp_bbox(defect.get("bbox", [0, 0, 1, 1]), width, height)
        if highlight:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), overlay_color, -1)
        thickness = max(2, int(round(min(width, height) / 220)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        name = str(defect.get("display_name") or DISPLAY_NAME)
        _draw_label(annotated, x1, y1, [name], box_color, label_bg)

    if highlight and result.get("defects"):
        annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
        for defect in result.get("defects", []):
            box_color, label_bg, _overlay = CLASS_COLORS.get(
                str(defect.get("class_name", "foreign_edge")),
                CLASS_COLORS["foreign_edge"],
            )
            x1, y1, x2, y2 = clamp_bbox(defect.get("bbox", [0, 0, 1, 1]), width, height)
            thickness = max(2, int(round(min(width, height) / 220)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
            name = str(defect.get("display_name") or DISPLAY_NAME)
            _draw_label(annotated, x1, y1, [name], box_color, label_bg)
    return _draw_software_trigger_hud(annotated, attach_trigger_metadata(result))


def attach_trigger_metadata(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(result or {})
    payload.setdefault("status", "PASS")
    payload.setdefault("defects", [])
    payload.setdefault("processing_time_ms", 0.0)
    now = datetime.now()
    payload.setdefault("timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))
    payload.setdefault("inspection_id", now.strftime("INS-%Y%m%d-%H%M%S"))
    return payload


def _draw_software_trigger_hud(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Stamp inspection ID, result, and timestamp on the output frame."""
    height, width = image.shape[:2]
    scale = max(0.45, min(width, height) / 720.0)
    bar_h = max(48, int(min(height * 0.14, 86)))
    y1 = max(0, height - bar_h)
    roi = image[y1:height, :]
    image[y1:height, :] = cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0)

    status = str(result.get("status", "PASS")).upper()
    fail = status == "FAIL"
    accent = (36, 36, 230) if fail else (46, 180, 72)
    cv2.rectangle(image, (0, y1), (width - 1, height - 1), accent, max(2, int(round(2 * scale))))
    stripe = max(5, int(9 * scale))
    cv2.rectangle(image, (0, y1), (stripe, height), accent, -1)

    defects = result.get("defects") or []
    defect_name = str(defects[0].get("display_name", "NONE")) if defects else "NONE"
    cycle = float(result.get("processing_time_ms") or 0.0)
    line1 = "INSPECTION"
    insp_id = str(result.get("inspection_id", ""))
    line1_full = f"{line1}   {insp_id}".strip()
    line2 = f"{status}   {defect_name}   {cycle:.0f} ms"
    stamp = str(result.get("timestamp", ""))

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 1.6)))
    font_scale = scale * 0.55
    pad = max(10, int(14 * scale))
    (id_w, _), _ = cv2.getTextSize(line1_full, font, font_scale, thickness)
    (rw, _), _ = cv2.getTextSize(stamp, font, font_scale * 0.92, thickness)
    if pad + stripe + id_w + rw + 24 > width:
        line1_full = line1
        line2 = f"{line2}   {insp_id}".strip()
    y_top = y1 + int(bar_h * 0.42)
    y_bot = y1 + int(bar_h * 0.82)
    cv2.putText(image, line1_full, (pad + stripe, y_top), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    (rw, _rh), _ = cv2.getTextSize(stamp, font, font_scale * 0.92, thickness)
    cv2.putText(
        image,
        stamp,
        (max(pad, width - rw - pad), y_top),
        font,
        font_scale * 0.92,
        (210, 216, 226),
        thickness,
        cv2.LINE_AA,
    )
    status_color = (70, 70, 255) if fail else (90, 220, 120)
    cv2.putText(
        image,
        line2,
        (pad + stripe, min(height - 6, y_bot)),
        font,
        font_scale * 1.08,
        status_color,
        thickness + 1,
        cv2.LINE_AA,
    )
    return image


def empty_result(processing_time_ms: float = 0.0) -> dict[str, Any]:
    return attach_trigger_metadata(
        {
            "status": "PASS",
            "defects": [],
            "processing_time_ms": float(processing_time_ms),
        }
    )
