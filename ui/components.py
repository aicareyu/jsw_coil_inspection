from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "care_yu_logo.png"


INDUSTRIAL_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap");

.stApp {
    background:
        radial-gradient(900px 420px at 8% -8%, rgba(245, 165, 36, 0.16), transparent 55%),
        radial-gradient(800px 480px at 92% 0%, rgba(56, 189, 248, 0.10), transparent 50%),
        linear-gradient(180deg, #101828 0%, #070b12 42%, #05070d 100%);
    color: #eef3fb;
    font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem; max-width: 1540px; }

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 22px;
    border: 1px solid rgba(245, 165, 36, 0.22);
    background: linear-gradient(115deg, rgba(18, 24, 38, 0.96) 0%, rgba(22, 32, 52, 0.92) 100%);
    border-radius: 16px;
    margin-bottom: 18px;
    gap: 18px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
    position: relative;
    overflow: hidden;
}
.app-header:before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    background: linear-gradient(180deg, #f5a524, #ef4444 70%, #38bdf8);
}
.brand {
    display: flex;
    align-items: center;
    gap: 16px;
    min-width: 0;
    padding-left: 8px;
}
.brand-logo {
    height: 74px;
    width: auto;
    background: #ffffff;
    border-radius: 12px;
    padding: 6px 8px;
    flex-shrink: 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.app-title {
    font-size: 1.42rem;
    letter-spacing: 0.06em;
    font-weight: 700;
    color: #fff8ee;
}
.app-subtitle {
    color: #9aa8bf;
    font-size: 0.8rem;
    letter-spacing: 0.14em;
    margin-top: 5px;
}
.status-row { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
    border-radius: 999px;
    padding: 7px 13px;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    border: 1px solid #334057;
    background: rgba(10, 16, 28, 0.72);
    color: #d5deec;
    font-weight: 600;
}
.pill.ready { border-color: rgba(74, 222, 128, 0.45); color: #4ade80; background: rgba(22, 101, 52, 0.28); }
.pill.warn { border-color: rgba(251, 191, 36, 0.45); color: #fbbf24; background: rgba(120, 53, 15, 0.28); }
.pill.bad { border-color: rgba(248, 113, 113, 0.5); color: #f87171; background: rgba(127, 29, 29, 0.28); }

.card {
    background: linear-gradient(180deg, rgba(22, 30, 46, 0.96), rgba(14, 20, 32, 0.96));
    border: 1px solid #2a3548;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
}
.image-empty {
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 10px;
    border: 1px dashed #3b4a66;
    background: repeating-linear-gradient(
        135deg,
        rgba(255,255,255,0.015) 0 12px,
        rgba(255,255,255,0.04) 12px 24px
    );
}
.result-box {
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    gap: 8px;
}
.result-box.fail-card {
    border-color: rgba(239, 68, 68, 0.45);
    box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.15), 0 18px 40px rgba(127, 29, 29, 0.25);
    background: linear-gradient(180deg, rgba(70, 18, 24, 0.55), rgba(16, 18, 28, 0.96));
}
.result-box.pass-card {
    border-color: rgba(34, 197, 94, 0.4);
    box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.12), 0 18px 40px rgba(20, 83, 45, 0.22);
    background: linear-gradient(180deg, rgba(16, 56, 36, 0.5), rgba(16, 18, 28, 0.96));
}
.result-box.ready-card {
    border-color: rgba(245, 165, 36, 0.4);
    box-shadow: 0 0 0 1px rgba(245, 165, 36, 0.12), 0 18px 40px rgba(120, 53, 15, 0.18);
}
.result-kicker {
    letter-spacing: 0.22em;
    color: #9aa8bf;
    font-size: 0.76rem;
    font-weight: 600;
}
.result-status {
    font-size: 5.4rem;
    line-height: 0.92;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-shadow: 0 10px 28px rgba(0,0,0,0.35);
}
.result-status.fail { color: #ff4d5a; }
.result-status.pass { color: #3ee08a; }
.result-defect {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #ffe4c4;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(245, 165, 36, 0.12);
    border: 1px solid rgba(245, 165, 36, 0.28);
}
.metric-grid {
    width: 100%;
    margin-top: 16px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    text-align: left;
}
.metric-tile {
    background: rgba(8, 12, 20, 0.45);
    border: 1px solid #2a3548;
    border-radius: 10px;
    padding: 8px 10px;
}
.metric-label { color: #8b97ab; font-size: 0.7rem; letter-spacing: 0.07em; }
.metric-value { color: #e8eef7; font-size: 0.92rem; font-weight: 600; word-break: break-all; }

.history-line {
    font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82rem;
    color: #d5deec;
    padding: 10px 12px;
    margin-bottom: 8px;
    border: 1px solid #243044;
    border-radius: 10px;
    background: rgba(16, 22, 34, 0.7);
}
.small-muted { color: #9aa8bf; font-size: 0.84rem; }
.panel-title {
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    color: #f5a524;
    font-weight: 700;
    margin-bottom: 8px;
}

div.stButton > button {
    background: linear-gradient(180deg, #2b3d63 0%, #1d2a46 100%);
    color: white;
    border: 1px solid #3b4d72;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.04em;
    min-height: 2.6rem;
}
div.stButton > button:hover { background: #2f4674; border-color: #5b73a3; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #f6b53d 0%, #e08912 100%);
    color: #1a1206;
    border: 0;
    box-shadow: 0 10px 22px rgba(245, 165, 36, 0.28);
}
div.stButton > button[kind="primary"]:hover { background: #ffc14d; }

[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(20, 28, 44, 0.9), rgba(12, 16, 26, 0.9));
    border: 1px dashed #f5a52466;
    border-radius: 14px;
    padding: 10px;
}
[data-testid="stImage"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #2a3548;
    box-shadow: 0 16px 34px rgba(0,0,0,0.28);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(10, 14, 22, 0.45);
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #b7c3d8;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    background: rgba(245, 165, 36, 0.16) !important;
    color: #ffe7b8 !important;
}
hr { border-color: #243044 !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(INDUSTRIAL_CSS, unsafe_allow_html=True)


def _logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_header(system_ready: bool, camera_status: str, model_info: dict[str, Any]) -> None:
    system_cls = "ready" if system_ready else "bad"
    cam_cls = "ready" if camera_status == "CAMERA CONNECTED" else "warn"
    model_cls = "ready" if model_info.get("status") == "LOADED" else "warn"
    logo_uri = _logo_data_uri()
    logo_html = (
        f'<img class="brand-logo" src="{logo_uri}" alt="Care Yu Automation" />'
        if logo_uri
        else ""
    )
    st.markdown(
        f"""
        <div class="app-header">
          <div class="brand">
            {logo_html}
            <div>
              <div class="app-title"> COIL SURFACE INSPECTION</div>
              <div class="app-subtitle">CARE YU AUTOMATION &nbsp;|&nbsp; FOREIGN EDGE · EDGE RUB · LOOSE COIL</div>
            </div>
          </div>
          <div class="status-row">
            <div class="pill {system_cls}">SYSTEM READY</div>
            <div class="pill {cam_cls}">{camera_status}</div>
            <div class="pill {model_cls}">MODEL {model_info.get('status', 'DEMO MODE')}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def render_result_card(result: dict[str, Any] | None) -> None:
    if not result:
        st.markdown(
            """
            <div class="card result-box">
              <div class="result-kicker">RESULT</div>
              <div class="app-title" style="font-size:1.4rem;letter-spacing:0.08em;">WAITING</div>
              <div class="small-muted">Upload a coil image, then click ANALYZE to inspect.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status = result.get("status", "PASS")
    defects = result.get("defects") or []
    top = defects[0] if defects else None
    time_ms = result.get("processing_time_ms", 0)
    stamp = str(result.get("timestamp", "—"))
    insp_id = str(result.get("inspection_id", "—"))
    if status == "FAIL" and top:
        bbox = top.get("bbox", [0, 0, 0, 0])
        bbox_text = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
        st.markdown(
            f"""
            <div class="card result-box fail-card">
              <div class="result-kicker">RESULT</div>
              <div class="result-status fail">FAIL</div>
              <div class="result-defect">{top.get('display_name', 'DEFECT')}</div>
              <div class="metric-grid">
                <div class="metric-tile"><div class="metric-label">DEFECT COUNT</div><div class="metric-value">{len(defects)}</div></div>
                <div class="metric-tile"><div class="metric-label">DETECTION TIME</div><div class="metric-value">{time_ms:.0f} ms</div></div>
                <div class="metric-tile"><div class="metric-label">BOUNDING BOX</div><div class="metric-value">{bbox_text}</div></div>
                <div class="metric-tile"><div class="metric-label">INSPECTION ID</div><div class="metric-value">{insp_id}</div></div>
                <div class="metric-tile"><div class="metric-label">TIMESTAMP</div><div class="metric-value">{stamp}</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="card result-box pass-card">
              <div class="result-kicker">RESULT</div>
              <div class="result-status pass">PASS</div>
              <div class="small-muted">No FOREIGN EDGE, EDGE RUB, or LOOSE COIL defect detected</div>
              <div class="metric-grid">
                <div class="metric-tile"><div class="metric-label">DEFECT COUNT</div><div class="metric-value">0</div></div>
                <div class="metric-tile"><div class="metric-label">DETECTION TIME</div><div class="metric-value">{time_ms:.0f} ms</div></div>
                <div class="metric-tile"><div class="metric-label">BOUNDING BOX</div><div class="metric-value">—</div></div>
                <div class="metric-tile"><div class="metric-label">INSPECTION ID</div><div class="metric-value">{insp_id}</div></div>
                <div class="metric-tile"><div class="metric-label">TIMESTAMP</div><div class="metric-value">{stamp}</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if result.get("error"):
        st.warning(result["error"])


def format_history_row(row: dict[str, Any]) -> str:
    stamp = str(row.get("timestamp", ""))[11:19] or str(row.get("timestamp", ""))
    name = row.get("image_name", "-")
    result = row.get("result", "-")
    defect = row.get("defect_name") or "-"
    return f"{stamp} | {name} | {result} | {defect}"
