import logging
import sys
import os

import streamlit as st


ROOT_PATH = os.path.abspath(os.path.join(__file__, "../../../"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def import_module():
    src_module = os.path.join(ROOT_PATH, "src")  # Add src to sys.path
    if src_module not in sys.path:
        sys.path.insert(0, src_module)

    from app import intro, realtime_det, upload_det

    global intro_func, realtime_det_func, upload_det_func
    intro_func = intro.main
    realtime_det_func = realtime_det.main
    upload_det_func = upload_det.main


def main():
    import_module()
    pname2func = {
        "Intro": intro_func,
        "Detection with File Upload": upload_det_func,
        "Detection with Webcam": realtime_det_func,
    }

    st.set_page_config(page_title="Dashboard - Video Anomaly Detection")
    page_name = st.sidebar.selectbox("Navigate Page", list(pname2func.keys()))
    pname2func[page_name]()


if __name__ == "__main__":
    main()
