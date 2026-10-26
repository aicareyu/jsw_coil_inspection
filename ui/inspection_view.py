from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from ui.components import bgr_to_pil, render_result_card

PREVIEW_WIDTH = 240
OUTPUT_WIDTH = 420


def _output_filename(result: dict[str, Any] | None, source_label: str) -> str:
    status = str((result or {}).get("status", "OUTPUT")).upper()
    defects = (result or {}).get("defects") or []
    defect = str(defects[0].get("display_name", "PASS")).replace(" ", "-") if defects else "PASS"
    insp_id = str((result or {}).get("inspection_id", "output"))
    stem = source_label.split(".")[0].replace(" ", "_")[:40]
    return f"{stem}_{status}_{defect}_{insp_id}.jpg"


def render_output_panel(
    annotated: np.ndarray | None,
    result: dict[str, Any] | None,
    source_label: str = "INSPECTION IMAGE",
    awaiting_analyze: bool = False,
) -> None:
    st.markdown('<div class="panel-title">OUTPUT</div>', unsafe_allow_html=True)
    if awaiting_analyze:
        st.markdown(
            """
            <div class="card result-box ready-card">
              <div class="result-kicker">READY</div>
              <div class="small-muted">Click ANALYZE under the uploaded image to see the result here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if annotated is None and result is None:
        st.markdown(
            """
            <div class="card image-empty">
              <div class="result-kicker">OUTPUT</div>
              <span class="small-muted">Upload an image on the left, then click ANALYZE.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if annotated is not None:
        st.caption(source_label)
        st.image(bgr_to_pil(annotated), width=OUTPUT_WIDTH, caption="Inspection output")
    render_result_card(result)
    if annotated is not None:
        ok, encoded = cv2.imencode(".jpg", annotated)
        if ok:
            st.download_button(
                "SAVE OUTPUT",
                data=encoded.tobytes(),
                file_name=_output_filename(result, source_label),
                mime="image/jpeg",
                use_container_width=True,
            )
    if st.button("NEW INSPECTION", use_container_width=True):
        for key in (
            "last_original",
            "last_annotated",
            "last_result",
            "last_source_name",
            "pending_images",
            "pending_upload_key",
            "video_summary",
            "video_index",
            "video_playing",
        ):
            st.session_state.pop(key, None)
        st.rerun()
