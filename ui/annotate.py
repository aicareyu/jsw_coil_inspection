from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from inspection.classes import CLASS_IDS, DEFECT_CLASSES, class_id, display_name
from inspection.config import project_root
from inspection.postprocessing import annotate_image
from ui.components import bgr_to_pil


def _to_yolo_line(class_name: str, bbox: list[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id(class_name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def render_annotate_panel() -> None:
    st.markdown("### ANNOTATE")
    st.caption(
        "Mark FOREIGN EDGE, EDGE RUB, or LOOSE COIL and export a YOLO label. "
        "This does not train a model. After enough labels, run: python -m scripts.train_yolo"
    )
    uploaded = st.file_uploader(
        "Upload image to annotate",
        type=["jpg", "jpeg", "png"],
        key="annotate_uploader",
    )
    if uploaded is None:
        st.info("Future path: Upload → Annotate → Export YOLO dataset → Train YOLO11 → Deploy models/best.pt")
        return

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("This file could not be read as an image.")
        return

    height, width = image.shape[:2]
    class_name = st.selectbox(
        "Defect class",
        options=list(DEFECT_CLASSES.keys()),
        format_func=lambda key: f"{CLASS_IDS[key]}  {DEFECT_CLASSES[key]}",
    )
    c1, c2, c3, c4 = st.columns(4)
    x1 = c1.number_input("x1", 0, width - 1, int(width * 0.25))
    y1 = c2.number_input("y1", 0, height - 1, int(height * 0.25))
    x2 = c3.number_input("x2", 1, width, int(width * 0.75))
    y2 = c4.number_input("y2", 1, height, int(height * 0.75))
    preview = annotate_image(
        image,
        {
            "status": "FAIL",
            "defects": [
                {
                    "class_name": class_name,
                    "display_name": display_name(class_name),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            ],
        },
    )
    st.image(bgr_to_pil(preview), use_column_width=True, caption="Annotation preview")

    split = st.selectbox("Dataset split", ["train", "val"])
    if st.button("EXPORT YOLO ANNOTATION", use_container_width=True):
        root = project_root()
        stem = f"{Path(uploaded.name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        image_dir = root / "dataset" / "images" / split
        label_dir = root / "dataset" / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        cv2.imwrite(str(image_path), image)
        label_path.write_text(
            _to_yolo_line(class_name, [int(x1), int(y1), int(x2), int(y2)], width, height),
            encoding="utf-8",
        )
        st.success(
            f"Saved {image_path.name} and {label_path.name}. "
            f"Class {class_id(class_name)} = {class_name}."
        )
