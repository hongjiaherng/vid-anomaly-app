import atexit
import logging
import tempfile
import os

import torch
import yt_dlp as youtube_dl
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.model_builder import load_backbone, load_detector
from app.detection_runner import run_detection

logger = logging.getLogger(__name__)


def cleanup_tempfile(temp_file: tempfile.NamedTemporaryFile):
    logger.info(f"Clean up {temp_file.name=}")
    temp_file.close()
    os.unlink(temp_file.name)


def init_session_state():
    if "temp_file" not in st.session_state:
        st.session_state.temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        atexit.register(cleanup_tempfile, st.session_state.temp_file)
        logger.info(f"Init global {st.session_state.temp_file.name=}")


def handle_upload(video_file: UploadedFile):
    temp_file = st.session_state.temp_file

    @st.cache_data(show_spinner=False)
    def _load_into_temp(video_file: UploadedFile):
        try:
            temp_file.seek(0)  # Reset file pointer (in case there's prevously uploaded data in there)
            temp_file.write(video_file.read())
            logger.info(f"Loaded {video_file.name=} to {temp_file.name=}")
            return True

        except Exception:
            logger.error(f"Failed to load {video_file.name=} to {temp_file.name=}")
            return False

    logger.info(f"Handling upload of {video_file.name if video_file else None}")

    # No video uploaded, clear temp file content, ensure the temp file is empty
    if video_file is None:
        temp_file.truncate(0)  # Clear file content, in case there's prevously uploaded file content in there
        logger.info(f"Cleared {temp_file.name=}")
        return

    # Video uploaded, load video to temp file
    upload_status = _load_into_temp(video_file)
    if upload_status:
        _, vid_container, _ = st.columns([0.15, 0.7, 0.15])
        vid_container.video(temp_file.name)
    else:
        st.error("Failed to upload video")


def handle_download(video_url: str):
    temp_file = st.session_state.temp_file

    @st.cache_data(show_spinner=False)
    def _load_into_temp(video_url: str):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    "format": "mp4",
                    "outtmpl": f"{temp_dir}/%(id)s.%(ext)s",
                }
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                    info_dict = ydl.extract_info(video_url, download=False)
                    video_id, video_ext = info_dict.get("id", None), info_dict.get("ext", None)
                    video_path = f"{temp_dir}/{video_id}.{video_ext}"

                    temp_file.seek(0)  # Reset file pointer (in case there's prevously uploaded data in there)
                    temp_file.write(open(video_path, "rb").read())

            logger.info(f"Downloaded {video_url=} to {temp_file.name=}")
            return True

        except Exception:
            logger.error(f"Failed to download {video_url=}")
            return False

    logger.info(f"Handling download {video_url=}")
    if video_url == "" or len(video_url) == 0:
        temp_file.truncate(0)  # Clear file content, in case there's prevously uploaded file content in there
        logger.info(f"Cleared {temp_file.name=}")
        return

    dl_placeholder = st.empty()
    with dl_placeholder, st.spinner("Downloading ..."):
        download_status = _load_into_temp(video_url)

    if download_status:
        _, vid_container, _ = st.columns([0.15, 0.7, 0.15])
        vid_container.video(temp_file.name)
    else:
        dl_placeholder.error("Failed to download video")


def init_device(enable_gpu: bool) -> torch.device:
    if not enable_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        st.toast("GPU is not available, reverting back to CPU", icon="⚠️")
        return torch.device("cpu")


def main(**kwargs):
    init_session_state()

    st.title("Video Anomaly Detection Dashboard")
    st.divider()

    st.sidebar.divider()
    st.sidebar.markdown("## Settings")

    # Model config
    with st.sidebar.expander("**Model Configuration**", expanded=True):
        feature_name = st.selectbox("Feature Extractor", ["I3D", "C3D", "Video Swin"], index=0)
        model_name = st.selectbox("Model", ["Sultani-Net", "HL-Net", "SVM Baseline"], index=0)
        ckpt_type = st.selectbox("Checkpoint Type", ["Best", "Last"], index=0)
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)

    with st.sidebar.expander("**Miscellaneous**", expanded=True):
        enable_gpu = st.toggle("Enable GPU", value=True)

    device = init_device(enable_gpu)
    backbone_model, video_preprocessor, clip_preprocessor, sampling_strategy = load_backbone(feature_name, device)
    detector_model = load_detector(model_name, feature_name, ckpt_type, device)

    # Upload video
    col1, col2 = st.columns([3, 1])
    with col1, st.expander("**Video Upload**", expanded=True):
        video_source = st.radio("Video Source", ["Upload", "YouTube"], index=0, key="video_source", horizontal=True)

        if video_source == "Upload":
            video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
            handle_upload(video_file)

        else:
            video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=<video_id>")
            handle_download(video_url)

    frame_placeholder = st.empty()
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.markdown("**Anomaly Score**")
    anomaly_score_placeholder = metric_col1.markdown("0")

    metric_col2.markdown("**Frame Rate**")
    frame_rate_placeholder = metric_col2.markdown("0")

    metric_col3.markdown("**Resolution**")
    resolution_placeholder = metric_col3.markdown("0")

    chart_placeholder = st.empty()

    if col2.button("Run Analysis", type="primary"):
        if st.session_state.temp_file.tell() == 0:
            st.toast("Please upload a video first.", icon="⚠️")
            return
        run_detection(
            video_path=st.session_state.temp_file.name,
            video_preprocessor=video_preprocessor,
            clip_preprocessor=clip_preprocessor,
            sampling_strategy=sampling_strategy,
            backbone_model=backbone_model,
            detector_model=detector_model,
            device=device,
            frame_placeholder=frame_placeholder,
            anomaly_score_placeholder=anomaly_score_placeholder,
            frame_rate_placeholder=frame_rate_placeholder,
            resolution_placeholder=resolution_placeholder,
            chart_placeholder=chart_placeholder,
        )
