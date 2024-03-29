from typing import Optional
import atexit
import logging
import os
import tempfile

import streamlit as st
import yt_dlp as youtube_dl
from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)


def cleanup_tempfile(temp_file: tempfile.NamedTemporaryFile):
    logger.info(f"Clean up {temp_file.name=}")
    temp_file.close()
    os.unlink(temp_file.name)


def init_tempfile_in_session_state():
    if "temp_file" not in st.session_state:
        st.session_state.temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        atexit.register(cleanup_tempfile, st.session_state.temp_file)
        logger.info(f"Init global {st.session_state.temp_file.name=}")


def handle_upload(video_file: UploadedFile) -> Optional[str]:
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

    return video_file.name


def handle_download(video_url: str) -> Optional[str]:
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

    return video_url
