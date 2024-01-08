import atexit
import logging
import os
import tempfile

import streamlit as st
import yt_dlp as youtube_dl
from streamlit.runtime.uploaded_file_manager import UploadedFile


def download_youtube_video(url: str, temp_file: tempfile.NamedTemporaryFile, logger: logging.Logger):
    logger.info(f"Download {url=} to {temp_file.name=}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                "format": "mp4",
                "outtmpl": f"{temp_dir}/%(id)s.%(ext)s",
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                info_dict = ydl.extract_info(url, download=False)
                video_id, video_ext = info_dict.get("id", None), info_dict.get("ext", None)
                video_path = f"{temp_dir}/{video_id}.{video_ext}"

                temp_file.seek(0)  # Reset file pointer (in case there's prevously uploaded data in there)
                temp_file.write(open(video_path, "rb").read())

        return True

    except Exception:
        logger.error(f"Failed to download {url=}")
        return False


def load_uploaded_video(video_file: UploadedFile, temp_file: tempfile.NamedTemporaryFile, logger: logging.Logger):
    logger.info(f"Load {video_file.name=} to {temp_file.name=}")
    try:
        temp_file.seek(0)  # Reset file pointer (in case there's prevously uploaded data in there)
        temp_file.write(video_file.read())

        return True

    except Exception:
        logger.error(f"Failed to load {video_file.name=}")
        return False


def cleanup_tempfile(temp_file: tempfile.NamedTemporaryFile, logger: logging.Logger):
    logger.info(f"Clean up {temp_file.name=}")
    temp_file.close()
    os.unlink(temp_file.name)


def handle_upload(video_file: UploadedFile, temp_file: tempfile.NamedTemporaryFile, logger: logging.Logger):
    if video_file is None:
        temp_file.truncate(0)  # Clear file content, in case there's prevously uploaded file content in there
        return

    upload_status = load_uploaded_video(video_file, temp_file, logger=logger)

    if upload_status:
        st.video(temp_file.name)
    else:
        st.error("Failed to upload video")


def handle_download(video_url: str, temp_file: tempfile.NamedTemporaryFile, logger: logging.Logger):
    if video_url == "" or len(video_url) == 0:
        temp_file.truncate(0)  # Clear file content, in case there's prevously uploaded file content in there
        return

    download_placeholder = st.empty()
    with download_placeholder, st.spinner("Downloading ..."):
        download_status = download_youtube_video(video_url, temp_file, logger=logger)

    if download_status:
        download_placeholder.video(temp_file.name)
    else:
        download_placeholder.error("Failed to download video")


def init_session_state(logger: logging.Logger):
    if "temp_file" not in st.session_state:
        st.session_state.temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        atexit.register(cleanup_tempfile, st.session_state.temp_file, logger=logger)
        logger.info(f"Init {st.session_state.temp_file.name=}")


def main(logger: logging.Logger, **kwargs):
    init_session_state(logger)

    st.title("Video Anomaly Detection Dashboard")
    st.divider()

    st.sidebar.divider()
    st.sidebar.markdown("## Settings")

    # Model config
    with st.sidebar.expander("Model Configuration", expanded=True):
        # st.divider()
        feature_name = st.selectbox(
            "Feature Extractor",
            ["I3D", "C3D", "Video Swin"],
            index=0,
        )
        model_name = st.selectbox(
            "Model",
            ["HL-Net", "Sultani's Net", "RTFM-Net", "SVM Baseline"],
            index=0,
        )
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)

    with st.sidebar.expander("Miscellaneous", expanded=True):
        # st.divider()
        enable_gpu = st.checkbox("Enable GPU", value=True)

    with st.expander("**Video Upload**", expanded=True):
        video_source = st.radio("Video Source", ["Upload", "YouTube"], index=0, key="video_source", horizontal=True)

        if video_source == "Upload":
            video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "asf", "m4v"])
            handle_upload(video_file, temp_file=st.session_state.temp_file, logger=logger)

        else:
            video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=<video_id>")
            handle_download(video_url, temp_file=st.session_state.temp_file, logger=logger)

    # Perform anomaly detection
