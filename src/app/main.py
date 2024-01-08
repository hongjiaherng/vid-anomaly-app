import logging

import streamlit as st

import intro
import realtime_det
import upload_det


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    pname2func = {
        "Intro": intro.main,
        "Detection with File Upload": upload_det.main,
        "Detection with Webcam": realtime_det.main,
    }

    page_name = st.sidebar.selectbox("Navigate Page", list(pname2func.keys()))
    pname2func[page_name](logger=logger)


if __name__ == "__main__":
    main()
