import logging
import time
from typing import Dict, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from streamlit.delta_generator import DeltaGenerator
from anomaly_detector.models.pengwu_net import PengWuNet
from anomaly_detector.models.sultani_net import SultaniNet
from anomaly_detector.models.svm_baseline import BaselineNet
from app.utils.common import configure_settings_sidebar, init_device
from app.utils.file_handler import handle_download, handle_upload, init_tempfile_in_session_state
from app.utils.model_builder import load_backbone, load_detector, predict_pipeline

logger = logging.getLogger(__name__)

DISPLAY_FRAME_INTERVAL = 8  # Display a frame every 4 frames


def run_detection(
    video_path: str,
    video_name: str,
    video_preprocessor: torch.nn.Module,
    clip_preprocessor: torch.nn.Module,
    sampling_strategy: Dict[str, int],
    backbone_model: torch.nn.Module,
    detector_model: Union[PengWuNet, SultaniNet, BaselineNet],
    device: torch.device,
    threshold: float,
    frame_placeholder: DeltaGenerator,
    progress_bar: DeltaGenerator,
    status_text: DeltaGenerator,
    chart_placeholder: DeltaGenerator,
    summary_placeholder: DeltaGenerator,
):
    logger.info(f"Running detection on {video_path=}")

    # Prepare video
    clip_iter = video_preprocessor({"path": video_path, "id": video_path})
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    overall_score = 0.0

    for clip_i, clip_dict in enumerate(clip_iter):  # clip_dict: {"inputs": (crop, T, C, H, W), ...}
        clip_dict = clip_preprocessor(clip_dict)
        clip_score = predict_pipeline(
            clip_dict=clip_dict,
            backbone=backbone_model,
            detector=detector_model,
            device=device,
        )
        overall_score += clip_score

        for frame_i in range(0, sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"], DISPLAY_FRAME_INTERVAL):
            if DISPLAY_FRAME_INTERVAL != 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_i * sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"] + frame_i)
            ret, frame = cap.read()
            if not ret:
                break

            frame_placeholder.image(frame, channels="BGR", caption=video_name, use_column_width=True)

            if clip_i == 0 and frame_i == 0:
                chart_placeholder.line_chart(
                    data=pd.DataFrame(
                        {
                            "frame": np.array([frame_i], dtype=int),
                            "score": np.array([clip_score], dtype=np.float32),
                            "threshold": np.array([threshold], dtype=np.float32),
                        },
                    ),
                    x="frame",
                    y=["score", "threshold"],
                )
            else:
                chart_placeholder.add_rows(
                    pd.DataFrame(
                        {
                            "frame": np.array([clip_i * sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"] + frame_i], dtype=int),
                            "score": np.array([clip_score], dtype=np.float32),
                            "threshold": np.array([threshold], dtype=np.float32),
                        },
                    )
                )

            progress_bar.progress((clip_i * sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"] + frame_i + 1) / total_frames)
            status_text.caption(f"{clip_i * sampling_strategy['sampling_rate'] * sampling_strategy['clip_len'] + frame_i + 1}/{total_frames}")

    # Just in case the last frame is not displayed
    progress_bar.progress(1.0)
    status_text.caption(f"{total_frames}/{total_frames}")

    overall_score /= clip_i + 1
    overall_score = round(overall_score, 4)
    # Determine the color of the overall score based on its value
    score_color = "green" if overall_score < threshold else "red"

    # Display summary
    summary_placeholder.markdown(
        f"""
        <hr>

        #### Summary
        Overall anomaly score: <span style='color:{score_color}'>{overall_score}</span>

        <br>

        ##### Video Information

        <table>
            <tr><td><strong>Video name</strong></td><td>{video_name}</td></tr>
            <tr><td><strong>Video duration</strong></td><td>{duration} s</td></tr>
            <tr><td><strong>Frame rate</strong></td><td>{frame_rate} fps</td></tr>
            <tr><td><strong>Number of frames</strong></td><td>{clip_i * sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"] + frame_i + 1} frames</td></tr>
            <tr><td><strong>Resolution</strong></td><td>{width}x{height}</td></tr>
        </table>
        <br><br>

        ##### Processing Information
        <table>
            <tr><td><strong>Sampling rate</strong></td><td>{sampling_strategy["sampling_rate"]}</td></tr>
            <tr><td><strong>Clip length</strong></td><td>{sampling_strategy["clip_len"]} frames</td></tr>
            <tr><td><strong>Number of clips</strong></td><td>{clip_i + 1} clips</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )
    cap.release()


def main(**kwargs):
    st.header("Video Anomaly Detection Dashboard")
    st.markdown("Run anomaly detection on your video. You can either upload a video or provide a YouTube URL.")
    st.divider()

    init_tempfile_in_session_state()
    model_name, feature_name, ckpt_type, threshold, enable_gpu = configure_settings_sidebar()
    device = init_device(enable_gpu)
    backbone_model, video_preprocessor, clip_preprocessor, sampling_strategy = load_backbone(feature_name, device)
    detector_model = load_detector(model_name, feature_name, ckpt_type, device)

    col1, col2 = st.columns([3, 1])

    # Upload video
    with col1, st.expander("**Video Upload**", expanded=True):
        video_source = st.radio("Video Source", ["Upload", "YouTube"], index=0, key="video_source", horizontal=True)

        if video_source == "Upload":
            video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
            video_name = handle_upload(video_file)

        else:
            video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=<video_id>")
            video_name = handle_download(video_url)

    run_btn = col2.button("Run Analysis", type="primary")

    st.divider()

    if run_btn:
        if st.session_state.temp_file.tell() == 0:
            st.toast("Please upload a video first.", icon="⚠️")
            return

        _, frame_container, _ = st.columns([0.15, 0.7, 0.15])
        frame_placeholder = frame_container.empty()

        progress_col1, progress_col2 = st.columns([0.9, 0.1])
        progress_bar = progress_col1.progress(0)
        status_text = progress_col2.empty()
        # st.write("\n")
        st.divider()

        st.markdown("#### Result")

        st.write("\n")

        chart_placeholder = st.empty()
        summary_placeholder = st.empty()

        run_detection(
            video_path=st.session_state.temp_file.name,
            video_name=video_name,
            video_preprocessor=video_preprocessor,
            clip_preprocessor=clip_preprocessor,
            sampling_strategy=sampling_strategy,
            backbone_model=backbone_model,
            detector_model=detector_model,
            device=device,
            threshold=threshold,
            frame_placeholder=frame_placeholder,
            progress_bar=progress_bar,
            status_text=status_text,
            chart_placeholder=chart_placeholder,
            summary_placeholder=summary_placeholder,
        )
