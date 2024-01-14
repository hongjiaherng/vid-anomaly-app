from typing import Tuple

import streamlit as st
import torch


def init_device(enable_gpu: bool) -> torch.device:
    if not enable_gpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        st.toast("GPU is not available, reverting back to CPU", icon="⚠️")
        return torch.device("cpu")


def configure_settings_sidebar() -> Tuple[str, str, str, float, bool]:
    st.sidebar.divider()
    st.sidebar.markdown("## Settings")
    # Model config
    with st.sidebar.expander("**Model Configuration**", expanded=True):
        model_name = st.selectbox("Model", ["Sultani-Net", "HL-Net", "SVM Baseline"], index=0)
        feature_name = st.selectbox("Feature Extractor", ["I3D", "C3D", "Video Swin"], index=0)
        ckpt_type = st.selectbox("Checkpoint Type", ["Best", "Last"], index=0)
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)

    with st.sidebar.expander("**Miscellaneous**", expanded=True):
        enable_gpu = st.toggle("Enable GPU", value=True)

    return model_name, feature_name, ckpt_type, threshold, enable_gpu
