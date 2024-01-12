import logging

import streamlit as st


def main(**kwargs):
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
            ["HL-Net", "Sultani-Net", "SVM Baseline"],
            index=0,
        )
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)

    with st.sidebar.expander("Miscellaneous", expanded=True):
        # st.divider()
        enable_gpu = st.checkbox("Enable GPU", value=True)
