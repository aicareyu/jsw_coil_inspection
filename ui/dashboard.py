from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from camera.camera_manager import CameraManager
from database.history import InspectionHistory
from inspection.config import load_config, resolve_path
from inspection.detector import get_detector
from inspection.classes import DEFECT_CLASSES, SAMPLE_FOLDERS, display_name
from scripts.extract_samples import extract_all_samples
from ui.annotate import render_annotate_panel
from ui.components import LOGO_PATH, bgr_to_pil, format_history_row, inject_css, render_header
from ui.inspection_view import PREVIEW_WIDTH, render_output_panel
from video.video_processor import VideoProcessor

IMAGE_TYPES = {"jpg", "jpeg", "png"}
VIDEO_TYPES = {"mp4", "avi"}


def _bytes_to_bgr(data: bytes) -> np.ndarray | None:
    try:
        pil = Image.open(BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return decoded


def _run_inspection(
    image: np.ndarray,
    source_name: str,
    detector,
    history: InspectionHistory,
    show_loading: bool = False,
) -> None:
    if show_loading:
        with st.spinner("Inspecting image..."):
            started = time.perf_counter()
            result, annotated = detector.inspect(image)
            leftover = 1.0 - (time.perf_counter() - started)
            if leftover > 0:
                time.sleep(leftover)
    else:
        result, annotated = detector.inspect(image)
    st.session_state.last_original = image
    st.session_state.last_annotated = annotated
    st.session_state.last_result = result
    st.session_state.last_source_name = source_name
    history.add_from_result(source_name, result)


def _list_samples(samples_dir: Path, class_name: str | None = None) -> list[Path]:
    folder = samples_dir / SAMPLE_FOLDERS[class_name] if class_name else samples_dir
    if not folder.exists():
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        files.extend(folder.glob(ext))
    return sorted({path.resolve(): path for path in files}.values(), key=lambda p: p.name)


def render_model_panel(detector) -> None:
    info = detector.model_info()
    st.markdown('<div class="panel-title">MODEL</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
          <div class="metric-label">MODEL</div>
          <div class="metric-value">{info['name']}</div>
          <br>
          <div class="metric-label">BACKEND</div>
          <div class="metric-value">{info['backend']}</div>
          <br>
          <div class="metric-label">MODEL FILE</div>
          <div class="metric-value">{info['model_file']}</div>
          <br>
          <div class="metric-label">STATUS</div>
          <div class="metric-value">{info['status']}</div>
          <p class="small-muted" style="margin-top:12px;">{info['detail']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Classes for training**")
    st.code("0 foreign_edge\n1 edge_rub\n2 loose_coil", language="text")
    if info["status"] != "LOADED":
        st.warning("DEMO MODE — TRAINED MODEL NOT INSTALLED")
        st.caption(
            "Annotate samples, then train with `python -m scripts.train_yolo`. "
            "Place the resulting weights at models/best.pt and restart."
        )


def render_history_panel(history: InspectionHistory) -> None:
    st.markdown('<div class="panel-title">INSPECTION HISTORY</div>', unsafe_allow_html=True)
    rows = history.list_recent(20)
    if not rows:
        st.caption("No inspections yet.")
        return
    for row in rows:
        st.markdown(f'<div class="history-line">{format_history_row(row)}</div>', unsafe_allow_html=True)


def render_samples_panel(samples_root: Path, detector, history: InspectionHistory) -> None:
    st.markdown('<div class="panel-title">DEMO SAMPLES</div>', unsafe_allow_html=True)
    st.caption("These images come from the supplied defective-coil Excel file and use the same detector as uploads.")
    for class_name, label in DEFECT_CLASSES.items():
        samples = _list_samples(samples_root, class_name)[:3]
        st.markdown(f"**{label}**")
        if not samples:
            st.info(f"No samples found in samples/{SAMPLE_FOLDERS[class_name]}/.")
            continue
        cols = st.columns(min(3, len(samples)))
        for idx, path in enumerate(samples):
            with cols[idx]:
                image = cv2.imread(str(path))
                if image is not None:
                    st.image(bgr_to_pil(image), width=PREVIEW_WIDTH, caption=path.name)
                if st.button(
                    f"{display_name(class_name)} Sample {idx + 1}",
                    key=f"sample_{class_name}_{idx}",
                    use_container_width=True,
                ):
                    if image is None:
                        st.error("Sample image could not be read.")
                    else:
                        _run_inspection(image, path.name, detector, history)
                        st.rerun()


def _queue_uploaded_images(images: list[tuple[str, np.ndarray]]) -> None:
    st.session_state.pending_images = images
    st.session_state.last_original = images[0][1]
    st.session_state.last_source_name = images[0][0]
    st.session_state.pop("last_annotated", None)
    st.session_state.pop("last_result", None)


def _analyze_pending(detector, history: InspectionHistory) -> None:
    pending = list(st.session_state.get("pending_images") or [])
    if not pending:
        st.warning("Upload an image first, then click ANALYZE.")
        return
    with st.spinner("Analyzing..."):
        started = time.perf_counter()
        for name, image in pending:
            _run_inspection(image, name, detector, history, show_loading=False)
        leftover = 1.0 - (time.perf_counter() - started)
        if leftover > 0:
            time.sleep(leftover)


def render_upload_panel(detector, history: InspectionHistory, config: dict) -> None:
    st.markdown('<div class="panel-title">UPLOAD</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload coil images or a video",
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        accept_multiple_files=True,
        help="Accepted: JPG, JPEG, PNG, MP4, AVI",
    )
    images: list[tuple[str, np.ndarray]] = []
    videos = []
    for file_obj in files or []:
        suffix = Path(file_obj.name).suffix.lower().lstrip(".")
        if suffix in IMAGE_TYPES:
            image = _bytes_to_bgr(file_obj.getvalue())
            if image is None:
                st.error(f"{file_obj.name} could not be read as an image.")
                continue
            images.append((file_obj.name, image))
        elif suffix in VIDEO_TYPES:
            videos.append(file_obj)
        else:
            st.error(f"Unsupported file type: {file_obj.name}")

    if videos and not images:
        _handle_video(videos[0], detector, history, config)
        return

    if images:
        upload_key = tuple(f"{name}-{image.shape}-{int(image.mean())}" for name, image in images)
        if st.session_state.get("pending_upload_key") != upload_key:
            st.session_state.pending_upload_key = upload_key
            _queue_uploaded_images(images)
        st.caption("Uploaded preview")
        preview_cols = st.columns(min(3, len(images)))
        for idx, (name, image) in enumerate(images):
            with preview_cols[idx % len(preview_cols)]:
                st.image(bgr_to_pil(image), width=PREVIEW_WIDTH, caption=name)

    if st.button("ANALYZE", type="primary", use_container_width=True, key="upload_analyze"):
        if not st.session_state.get("pending_images"):
            st.warning("Upload an image first, then click ANALYZE.")
        else:
            _analyze_pending(detector, history)
            st.rerun()


def _handle_video(file_obj, detector, history: InspectionHistory, config: dict) -> None:
    video_cfg = config.get("video", {})
    tmp_dir = resolve_path("data")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_dir / f"upload_{file_obj.name}"
    video_path.write_bytes(file_obj.getvalue())

    c1, c2, c3 = st.columns(3)
    start = c1.button("Start Inspection", use_container_width=True)
    pause = c2.button("Pause", use_container_width=True)
    stop = c3.button("Stop", use_container_width=True)

    if start:
        processor = VideoProcessor(
            detector,
            frame_interval=int(video_cfg.get("frame_interval", 8)),
            max_frames=int(video_cfg.get("max_frames", 180)),
        )
        with st.spinner("Inspecting video frames..."):
            summary = processor.inspect(video_path)
        st.session_state.video_summary = summary
        st.session_state.video_index = 0
        st.session_state.video_playing = True
        if summary.frames:
            first = summary.frames[0]
            history.add_from_result(file_obj.name, {
                "status": summary.overall_result,
                "defects": first.get("defects", []),
                "processing_time_ms": first.get("processing_time_ms", 0),
            })
        if summary.error:
            st.error(summary.error)

    if pause:
        st.session_state.video_playing = False
    if stop:
        st.session_state.video_playing = False
        st.session_state.pop("video_summary", None)
        st.session_state.pop("video_index", None)
        return

    summary = st.session_state.get("video_summary")
    if not summary or not summary.frames:
        return

    idx = int(st.session_state.get("video_index", 0))
    idx = max(0, min(idx, len(summary.frames) - 1))
    frame = summary.frames[idx]
    st.session_state.last_original = frame["original"]
    st.session_state.last_annotated = frame["annotated"]
    st.session_state.last_result = {
        "status": frame["status"],
        "defects": frame["defects"],
        "processing_time_ms": frame["processing_time_ms"],
    }
    st.session_state.last_source_name = f"{file_obj.name} frame {frame['frame_number']}"

    st.markdown(
        f"**Frame {frame['frame_number']}** &nbsp;|&nbsp; "
        f"Status: **{frame['status']}** &nbsp;|&nbsp; "
        f"Detections on this frame: **{frame['defect_count']}**"
    )
    new_idx = st.slider("Inspected frame", 0, len(summary.frames) - 1, idx)
    if new_idx != idx:
        st.session_state.video_index = new_idx
        st.rerun()

    st.markdown("#### VIDEO SUMMARY")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Frames inspected", summary.total_frames_inspected)
    s2.metric("Defective frames", summary.defective_frames)
    s3.metric("Defect detections", summary.foreign_edge_detections)
    s4.metric("Detection %", f"{summary.detection_percentage:.1f}%")
    s5.metric("Overall result", summary.overall_result)


def render_camera_panel(detector, history: InspectionHistory, config: dict, camera: CameraManager) -> None:
    st.markdown('<div class="panel-title">LIVE CAMERA</div>', unsafe_allow_html=True)
    cam_cfg = config.get("camera", {})
    camera.camera_index = int(cam_cfg.get("camera_index", 0))
    c1, c2, c3 = st.columns(3)
    if c1.button("Start Camera", use_container_width=True):
        if camera.start():
            st.session_state.camera_on = True
            st.success("CAMERA CONNECTED")
        else:
            st.session_state.camera_on = False
            st.warning(camera.status)
            if camera.last_error:
                st.caption(camera.last_error)
    if c2.button("Stop Camera", use_container_width=True):
        camera.stop()
        st.session_state.camera_on = False
    capture_clicked = c3.button("Capture & Inspect", use_container_width=True)

    live_frame = None
    if st.session_state.get("camera_on"):
        ok, frame = camera.read()
        if ok and frame is not None:
            live_frame = frame
            st.session_state.last_camera_frame = frame
            st.image(bgr_to_pil(frame), use_column_width=True, caption="Live camera feed")
            st.caption("CAMERA CONNECTED")
        else:
            st.warning("CAMERA NOT AVAILABLE")
            camera.stop()
            st.session_state.camera_on = False

    st.caption("Browser camera fallback if a USB / laptop camera is not available:")
    browser_shot = st.camera_input("Use browser camera")

    target = None
    source_name = "camera_capture.jpg"
    if capture_clicked:
        target = live_frame if live_frame is not None else st.session_state.get("last_camera_frame")
        if target is None:
            st.warning("No camera frame is available to capture.")
    if browser_shot is not None:
        decoded = _bytes_to_bgr(browser_shot.getvalue())
        if decoded is not None and (
            st.session_state.get("last_browser_shot") != browser_shot.getvalue()
        ):
            target = decoded
            source_name = "browser_camera.jpg"
            st.session_state.last_browser_shot = browser_shot.getvalue()

    if target is not None:
        _run_inspection(target, source_name, detector, history)
        st.rerun()


def render_app() -> None:
    st.set_page_config(
        page_title="Coil Surface Inspection",
        page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "▣",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    config = load_config()
    detector = get_detector()
    history = InspectionHistory(
        resolve_path(config.get("paths", {}).get("history_db", "data/inspection_history.db")),
        max_rows=int(config.get("history", {}).get("max_rows", 200)),
    )
    if "camera_manager" not in st.session_state:
        cam_cfg = config.get("camera", {})
        st.session_state.camera_manager = CameraManager(
            camera_index=int(cam_cfg.get("camera_index", 0)),
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
        )
    camera: CameraManager = st.session_state.camera_manager

    samples_dir = resolve_path(config.get("paths", {}).get("samples_dir", "samples"))
    have_any = any(_list_samples(samples_dir, name) for name in SAMPLE_FOLDERS)
    if not have_any:
        extract_all_samples()

    camera_status = camera.status if st.session_state.get("camera_on") else "CAMERA NOT AVAILABLE"
    render_header(True, camera_status, detector.model_info())

    awaiting_analyze = bool(st.session_state.get("pending_images")) and (
        st.session_state.get("last_annotated") is None
    )
    left, right = st.columns([0.92, 1.18], gap="large")
    with left:
        tabs = st.tabs(
            ["IMAGE / VIDEO", "LIVE CAMERA", "DEMO SAMPLES", "HISTORY", "MODEL", "ANNOTATE"]
        )
        with tabs[0]:
            render_upload_panel(detector, history, config)
        with tabs[1]:
            render_camera_panel(detector, history, config, camera)
        with tabs[2]:
            render_samples_panel(samples_dir, detector, history)
        with tabs[3]:
            render_history_panel(history)
        with tabs[4]:
            render_model_panel(detector)
        with tabs[5]:
            render_annotate_panel()
    with right:
        render_output_panel(
            st.session_state.get("last_annotated"),
            st.session_state.get("last_result"),
            source_label=str(st.session_state.get("last_source_name", "INSPECTION IMAGE")).upper(),
            awaiting_analyze=awaiting_analyze,
        )
