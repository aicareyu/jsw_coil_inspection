from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from inspection.classes import display_name
from inspection.feature_classifier import DemoFeatureClassifier
from inspection.postprocessing import scale_result_to_original
from inspection.preprocessing import preprocess


class DemoForeignEdgeDetector:
    """Classical CV fallback for coil-edge defects.

    Localizes abnormal winding/edge regions, then classifies among
    FOREIGN EDGE, EDGE RUB, and LOOSE COIL. Isolated so it can be
    replaced by a trained YOLO model.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        demo = (config or {}).get("demo_detector", {})
        infer = (config or {}).get("inference", {})
        self.max_side = int(infer.get("max_side", 960))
        self.clahe_clip = float(demo.get("clahe_clip", 2.0))
        self.denoise_h = float(demo.get("denoise_h", 4.0))
        self.min_coil_coverage = float(demo.get("min_coil_coverage", 0.10))
        self.min_anisotropy = float(demo.get("min_anisotropy", 0.20))
        self.min_edge_energy = float(demo.get("min_edge_energy", 8.0))
        self.max_mean_saturation = float(demo.get("max_mean_saturation", 78.0))
        self.w_resid = float(demo.get("heat_residual_weight", 0.60))
        self.w_disorder = float(demo.get("heat_disorder_weight", 0.40))
        self.peak_threshold = float(demo.get("peak_threshold", 0.78))
        self.score_threshold = float(demo.get("score_threshold", 0.88))
        self.min_box_w = int(demo.get("min_box_w", 14))
        self.min_box_h = int(demo.get("min_box_h", 10))
        self.min_box_area = int(demo.get("min_box_area", 180))
        self.max_box_area_ratio = float(demo.get("max_box_area_ratio", 0.35))
        self.box_padding = int(demo.get("box_padding", 8))
        self.max_detections = int(demo.get("max_detections", 3))
        self.classifier = DemoFeatureClassifier()
        if not self.classifier.loaded:
            self._try_auto_train()

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            prepared = preprocess(
                image,
                max_side=self.max_side,
                clahe_clip=self.clahe_clip,
                denoise_h=self.denoise_h,
            )
        except Exception:
            return self._finish("PASS", [], started)

        processed = prepared.processed_image
        enhanced = prepared.enhanced
        mask, coverage = self._coil_mask(enhanced)
        if coverage < self.min_coil_coverage:
            return self._finish("PASS", [], started, prepared)

        if self._looks_non_metallic(processed, mask):
            return self._finish("PASS", [], started, prepared)

        coherence, energy, angle = self._anisotropy_maps(enhanced)
        roi = mask > 0
        if not np.any(roi):
            return self._finish("PASS", [], started, prepared)

        mean_coh = float(coherence[roi].mean())
        mean_energy = float(energy[roi].mean())
        if mean_coh < self.min_anisotropy or mean_energy < self.min_edge_energy:
            return self._finish("PASS", [], started, prepared)

        residual = self._directional_residual(enhanced, angle, energy, mask)
        disorder = self._local_disorder(angle, energy, mask)
        gap_map = self._gap_map(enhanced, mask)
        heat = self.w_resid * self._normalize(residual, mask)
        heat += self.w_disorder * self._normalize(disorder, mask)
        heat += 0.25 * self._normalize(gap_map, mask)
        heat[mask == 0] = 0
        red_mask = prepared.extras.get("red_mask")
        if red_mask is not None and float(red_mask.mean()) > 0.2:
            heat[red_mask > 0] *= 0.4
        # Specular flashes are common on coil edges and are not the defect.
        heat[enhanced > 220] *= 0.35

        boxes = self._localize(heat, mask)
        if not boxes:
            boxes = self._localize(self._normalize(gap_map, mask), mask)
        scene = self._scene_scores(enhanced, processed, mask, gap_map)
        if not self.classifier.loaded:
            self.classifier.load()
        class_name = self._classify(
            prepared.original_image,
            enhanced,
            residual,
            gap_map,
            boxes[0][2] if boxes else [0, 0, 1, 1],
            scene,
        )
        if class_name == "loose_coil":
            loose_boxes = self._localize_loose_coil(enhanced, mask)
            if loose_boxes:
                boxes = loose_boxes
        defects = []
        for score, peak, bbox in boxes:
            confidence = 1.0 / (1.0 + np.exp(-(score - 0.95) * 3.0))
            confidence = float(np.clip(0.52 + 0.44 * confidence, 0.52, 0.97))
            defects.append(
                {
                    "class_name": class_name,
                    "display_name": display_name(class_name),
                    "confidence": round(confidence, 3),
                    "bbox": bbox,
                    "score": round(float(score), 3),
                }
            )
        status = "FAIL" if defects else "PASS"
        result = self._finish(status, defects, started, prepared)
        return scale_result_to_original(
            result,
            processed.shape[:2],
            prepared.original_image.shape[:2],
        )

    def _finish(
        self,
        status: str,
        defects: list[dict[str, Any]],
        started: float,
        prepared=None,
    ) -> dict[str, Any]:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        payload: dict[str, Any] = {
            "status": status,
            "defects": defects,
            "processing_time_ms": round(elapsed_ms, 1),
            "backend": "demo_cv",
        }
        if prepared is not None:
            payload["processed_shape"] = prepared.processed_image.shape[:2]
        return payload

    def _looks_non_metallic(self, bgr: np.ndarray, mask: np.ndarray) -> bool:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        roi = mask > 0
        if not np.any(roi):
            return True
        return float(sat[roi].mean()) > self.max_mean_saturation

    def _coil_mask(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Keep metallic mid-tones; drop blown highlights such as coil-eye sky.
        candidate = (blur > 22) & (blur < 210)
        mean = cv2.blur(gray.astype(np.float32), (15, 15))
        sq = cv2.blur(np.square(gray.astype(np.float32)), (15, 15))
        std = np.sqrt(np.maximum(sq - np.square(mean), 0))
        candidate = candidate & (std > 5)
        mask = (candidate.astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return np.zeros_like(gray, dtype=np.uint8), 0.0
        keep = np.zeros_like(mask)
        min_area = gray.size * 0.04
        for idx in range(1, count):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == idx] = 255
        if keep.mean() < 1:
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep[labels == idx] = 255
        return keep, float(keep.mean() / 255.0)

    def _anisotropy_maps(self, gray: np.ndarray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ixx = cv2.GaussianBlur(gx * gx, (0, 0), 2.2)
        iyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.2)
        ixy = cv2.GaussianBlur(gx * gy, (0, 0), 2.2)
        delta = np.sqrt(np.maximum(np.square(ixx - iyy) + 4.0 * np.square(ixy), 0))
        lam1 = 0.5 * (ixx + iyy + delta)
        lam2 = 0.5 * (ixx + iyy - delta)
        coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-6)
        energy = np.sqrt(gx * gx + gy * gy)
        angle = np.arctan2(gy, gx)
        return coherence, energy, angle

    def _directional_residual(
        self,
        gray: np.ndarray,
        angle: np.ndarray,
        energy: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        roi = mask > 0
        if int(roi.sum()) < 100:
            return np.zeros_like(gray, dtype=np.float32)
        hist, bins = np.histogram(
            angle[roi], bins=36, range=(-np.pi, np.pi), weights=energy[roi]
        )
        peak = int(np.argmax(hist))
        dominant = 0.5 * (bins[peak] + bins[peak + 1])
        wind_deg = float(np.degrees(dominant + np.pi / 2.0))
        length = max(17, int(min(gray.shape) / 10))
        if length % 2 == 0:
            length += 1
        kernel = np.zeros((length, length), dtype=np.float32)
        cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1, 1)
        rot = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), -wind_deg, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (length, length))
        kernel /= kernel.sum() + 1e-6
        smooth = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        residual = np.abs(gray.astype(np.float32) - smooth)
        residual[mask == 0] = 0
        return residual

    def _local_disorder(
        self, angle: np.ndarray, energy: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        cos2 = cv2.GaussianBlur(np.cos(2.0 * angle) * energy, (0, 0), 2.5)
        sin2 = cv2.GaussianBlur(np.sin(2.0 * angle) * energy, (0, 0), 2.5)
        energy_blur = cv2.GaussianBlur(energy, (0, 0), 2.5) + 1e-6
        alignment = np.sqrt(cos2 * cos2 + sin2 * sin2) / energy_blur
        disorder = (1.0 - alignment) * energy
        disorder[mask == 0] = 0
        return disorder

    def _normalize(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        roi = values[mask > 0]
        if roi.size == 0:
            return np.zeros_like(values, dtype=np.float32)
        p92 = float(np.percentile(roi, 92)) + 1e-6
        return np.clip(values / p92, 0, 2.5).astype(np.float32)

    def _border_touches(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> int:
        margin = 3
        count = 0
        if x1 <= margin:
            count += 1
        if y1 <= margin:
            count += 1
        if x2 >= width - 1 - margin:
            count += 1
        if y2 >= height - 1 - margin:
            count += 1
        return count

    def _valid_box(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> bool:
        box_w, box_h = x2 - x1, y2 - y1
        area = box_w * box_h
        if box_w < self.min_box_w or box_h < self.min_box_h:
            return False
        if area < self.min_box_area:
            return False
        if area > self.max_box_area_ratio * width * height:
            return False
        if max(box_w / width, box_h / height) > 0.92:
            return False
        touches = self._border_touches(x1, y1, x2, y2, width, height)
        if touches >= 3:
            return False
        if (box_h > 0.50 * height and box_w < 0.14 * width) or (
            box_w > 0.50 * width and box_h < 0.14 * height
        ):
            return False
        if touches >= 1 and min(box_w / width, box_h / height) < 0.10 and max(
            box_w / width, box_h / height
        ) > 0.40:
            return False
        if touches >= 2 and min(box_w, box_h) < 18:
            return False
        return True

    def _iou(self, a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union else 0.0

    def _localize(self, heat: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
        blurred = cv2.GaussianBlur(heat, (0, 0), 2.0)
        height, width = heat.shape
        dilated = cv2.dilate(blurred, np.ones((21, 21), np.uint8))
        peaks = (blurred >= dilated) & (blurred > 0.85) & (mask > 0)
        ys, xs = np.where(peaks)
        if xs.size == 0:
            yy, xx = np.unravel_index(int(np.argmax(blurred)), blurred.shape)
            ys, xs = np.array([yy]), np.array([xx])

        candidates: list[tuple[float, float, list[int]]] = []
        used = np.zeros_like(mask)
        order = np.argsort(blurred[ys, xs])[::-1]
        for index in order[:12]:
            y0, x0 = int(ys[index]), int(xs[index])
            if used[y0, x0]:
                continue
            peak = float(blurred[y0, x0])
            if peak < self.peak_threshold:
                continue
            local_thr = max(0.55, peak * 0.55)
            seeds = ((blurred >= local_thr) & (mask > 0)).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds)
            if count <= 1:
                continue
            label_id = int(labels[y0, x0])
            if label_id == 0:
                continue
            core = (labels == label_id) & (blurred >= max(local_thr, peak * 0.62))
            ys2, xs2 = np.where(core)
            if xs2.size < 30:
                ys2, xs2 = np.where(labels == label_id)
            if xs2.size < 20:
                continue
            x1, x2 = int(xs2.min()), int(xs2.max()) + 1
            y1, y2 = int(ys2.min()), int(ys2.max()) + 1
            pad = self.box_padding
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if not self._valid_box(x1, y1, x2, y2, width, height):
                continue
            region = blurred[y1:y2, x1:x2]
            score = float(region.mean() + 0.35 * peak)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            interior = min(cx / width, 1.0 - cx / width, cy / height, 1.0 - cy / height)
            score += 0.20 * interior
            score -= 0.18 * self._border_touches(x1, y1, x2, y2, width, height)
            if score < self.score_threshold:
                used[labels == label_id] = 1
                continue
            candidates.append((score, peak, [x1, y1, x2, y2]))
            used[labels == label_id] = 1

        candidates.sort(key=lambda item: item[0], reverse=True)
        kept: list[tuple[float, float, list[int]]] = []
        for item in candidates:
            if all(self._iou(item[2], existing[2]) < 0.45 for existing in kept):
                kept.append(item)
            if len(kept) >= self.max_detections:
                break
        return kept

    def _localize_loose_coil(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> list[tuple[float, float, list[int]]]:
        """Box the full coil face — looseness is a whole-wrap defect."""
        height, width = gray.shape
        min_dim = min(height, width)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(1, min_dim // 2),
            param1=80,
            param2=40,
            minRadius=max(30, min_dim // 5),
            maxRadius=max(31, min_dim // 2),
        )
        if circles is not None:
            cx, cy, radius = max(circles[0], key=lambda item: float(item[2]))
            cx, cy, radius = float(cx), float(cy), float(radius)
            centered = (0.22 * width < cx < 0.78 * width) and (
                0.18 * height < cy < 0.78 * height
            )
            if centered and radius >= 0.28 * min_dim:
                pad = int(max(6, 0.02 * radius))
                x1 = max(0, int(round(cx - radius)) - pad)
                y1 = max(0, int(round(cy - radius)) - pad)
                x2 = min(width, int(round(cx + radius)) + pad)
                y2 = min(height, int(round(cy + radius)) + pad)
                if x2 - x1 >= 40 and y2 - y1 >= 40:
                    return [(1.6, 1.6, [x1, y1, x2, y2])]

        coil = mask.copy()
        stand_y = int(0.88 * height)
        coil[stand_y:, :] = 0
        ys, xs = np.where(coil > 0)
        if xs.size < 200:
            ys, xs = np.where(mask > 0)
        if xs.size < 200:
            return []
        pad = 8
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(width, int(xs.max()) + pad)
        y2 = min(stand_y, int(ys.max()) + pad)
        if x2 - x1 < 40 or y2 - y1 < 40:
            return []
        return [(1.5, 1.5, [x1, y1, x2, y2])]

    def _gap_map(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        dark = cv2.GaussianBlur(gray, (5, 5), 0)
        inverted = cv2.bitwise_not(dark)
        gaps = cv2.normalize(blackhat.astype(np.float32) + inverted.astype(np.float32) * 0.25, None, 0, 1, cv2.NORM_MINMAX)
        gaps[mask == 0] = 0
        return gaps.astype(np.float32)

    def _scene_scores(
        self,
        enhanced: np.ndarray,
        bgr: np.ndarray,
        mask: np.ndarray,
        gap_map: np.ndarray,
    ) -> dict[str, float]:
        roi = mask > 0
        if not np.any(roi):
            return {"edge_density": 0.0, "gap_score": 0.0, "background": 1.0, "strap": 0.0}
        edges = cv2.Canny(enhanced, 40, 120)
        edge_density = float(edges[roi].mean() / 255.0)
        gap_score = float(gap_map[roi].mean())
        background = 1.0 - float(mask.mean() / 255.0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        dark_band = (hsv[:, :, 2] < 45) & (mask == 0)
        strap = float(dark_band.mean())
        return {
            "edge_density": edge_density,
            "gap_score": gap_score,
            "background": background,
            "strap": strap,
        }

    def _try_auto_train(self) -> None:
        try:
            from scripts.train_demo_classifier import train_demo_classifier

            train_demo_classifier()
            self.classifier.load()
        except Exception:
            return

    def _classify(
        self,
        processed: np.ndarray,
        enhanced: np.ndarray,
        residual: np.ndarray,
        gap_map: np.ndarray,
        bbox: list[int],
        scene: dict[str, float],
    ) -> str:
        if self.classifier.loaded or self.classifier.load():
            return self.classifier.predict(processed)
        x1, y1, x2, y2 = bbox
        height, width = enhanced.shape[:2]
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(max(width * height, 1))
        region_res = residual[y1:y2, x1:x2]
        region_gap = gap_map[y1:y2, x1:x2]
        sharp = float(cv2.Laplacian(enhanced[y1:y2, x1:x2], cv2.CV_32F).var()) if region_res.size else 0.0
        elong = max(x2 - x1, y2 - y1) / float(max(min(x2 - x1, y2 - y1), 1))

        loose = 0.0
        loose += 2.4 * scene["gap_score"]
        loose += 1.6 * scene["background"]
        loose += 1.8 * scene["strap"]
        loose += 0.8 * float(region_gap.mean()) if region_gap.size else 0.0
        if scene["edge_density"] < 0.12:
            loose += 1.1
        if area_ratio > 0.12:
            loose += 0.6

        rub = 0.0
        rub += 1.3 * min(area_ratio * 8.0, 1.4)
        rub += 0.9 * min(elong / 4.0, 1.2)
        if 0.12 <= scene["edge_density"] <= 0.32:
            rub += 0.8
        if sharp < 280:
            rub += 0.7
        if scene["strap"] < 0.02 and scene["background"] < 0.45:
            rub += 0.4

        foreign = 0.0
        foreign += 1.2 * min(scene["edge_density"] * 5.0, 1.5)
        if sharp > 220:
            foreign += 0.9
        if area_ratio < 0.10:
            foreign += 0.8
        if scene["gap_score"] < 0.18:
            foreign += 0.5
        if scene["strap"] < 0.015:
            foreign += 0.3

        scores = {"loose_coil": loose, "edge_rub": rub, "foreign_edge": foreign}
        return max(scores, key=scores.get)
