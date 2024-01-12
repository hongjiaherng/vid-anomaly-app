# TODO: Format anomaly score to 2 decimal places
# TODO: Calculate frame rate
# TODO: Center align anomaly score, frame rate, resolution

# TODO: Add result section
# TODO: Add number of frames
# TODO: Add number of clips
# TODO: Add sampling rate and clip length
# TODO: Add overall anomaly score

# TODO: Add divider between them? to make things more organized
# TODO: Add video filename if uploaded, video url if downloaded

# TODO: Try make it smoother by adding time.sleep(0.1) after each frame
# TODO: Draw theshold line on chart


import logging
from typing import Dict, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from anomaly_detector.models.pengwu_net import PengWuNet
from anomaly_detector.models.sultani_net import SultaniNet
from anomaly_detector.models.svm_baseline import BaselineNet

logger = logging.getLogger(__name__)


def predict_pipeline(
    clip_dict: Dict[str, torch.Tensor],
    backbone: torch.nn.Module,
    detector: Union[PengWuNet, SultaniNet, BaselineNet],
    sampling_strategy: Dict[str, int],
    device: torch.device,
) -> np.ndarray:
    assert (
        "sampling_rate" in sampling_strategy and "clip_len" in sampling_strategy
    ), "sampling_strategy must contain 'sampling_rate' and 'clip_len' keys"
    assert "inputs" in clip_dict, "clip_dict must contain 'inputs' key"

    if isinstance(detector, PengWuNet):  # PengWuNet: (B, T=1, D) -> (B, T=1, 1)
        clip_in = clip_dict["inputs"].squeeze(0).to(device)
        with torch.no_grad():
            clip_emb = backbone(clip_in)  # (B, D)
            clip_emb = clip_emb.unsqueeze(1)  # (B, 1, D) # PengWuNet expects (B, T, D) where T=1 for online inference
            clip_score = detector.predict(inputs=clip_emb, online=True)  # (B, T=1, 1)
            clip_score = torch.mean(clip_score).cpu().numpy()  # (1,) -> (T * sampling_rate * clip_len,)
        return clip_score.repeat(sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"])

    elif isinstance(detector, SultaniNet) or isinstance(detector, BaselineNet):
        clip_in = clip_dict["inputs"].squeeze(0).to(device)
        with torch.no_grad():
            clip_emb = backbone(clip_in)  # (B, D)
            clip_score = detector.predict(inputs=clip_emb)  # (B, 1)
            clip_score = torch.mean(clip_score).cpu().numpy()
        return clip_score.repeat(sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"])  # (1,) -> (T * sampling_rate * clip_len,)

    else:
        raise ValueError(f"Unknown detector type {type(detector)}")


def run_detection(
    video_path: str,
    video_preprocessor: torch.nn.Module,
    clip_preprocessor: torch.nn.Module,
    sampling_strategy: Dict[str, int],
    backbone_model: torch.nn.Module,
    detector_model: Union[PengWuNet, SultaniNet, BaselineNet],
    device: torch.device,
    frame_placeholder: any,
    anomaly_score_placeholder: any,
    frame_rate_placeholder: any,
    resolution_placeholder: any,
    chart_placeholder: any,
):
    logger.info(f"Running detection on {video_path=}")

    # Init streamlit components
    chart_placeholder.line_chart(
        pd.DataFrame({"Anomaly Score": np.array([0.0], dtype=np.float32), "Frame Index": np.array([0], dtype=int)}),
        x="Frame Index",
        y="Anomaly Score",
    )  # Initialize chart
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Prepare video
    clip_iter = video_preprocessor({"path": video_path, "id": video_path})
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"]

    for clip_i, clip_dict in enumerate(clip_iter):  # clip_dict: {"inputs": (crop, T, C, H, W), ...}
        clip_dict = clip_preprocessor(clip_dict)
        clip_score = predict_pipeline(
            clip_dict=clip_dict,
            backbone=backbone_model,
            detector=detector_model,
            sampling_strategy=sampling_strategy,
            device=device,
        )  # (T * sampling_rate * clip_len,)

        # Now we have clip_score (sampling_rate * clip_len = 2 * 32,) of clip i, we need to display them frame by frame
        for frame_i in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break

            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            anomaly_score_placeholder.markdown(clip_score[frame_i])
            resolution_placeholder.markdown(f"{frame.shape[1]}x{frame.shape[0]}")
            chart_placeholder.add_rows(
                pd.DataFrame(
                    {
                        "Anomaly Score": np.array([clip_score[frame_i]], dtype=np.float32),
                        "Frame Index": np.array([clip_i * frames_per_clip + frame_i + 1], dtype=int),
                    }
                )
            )
            progress_bar.progress((clip_i * frames_per_clip + frame_i + 1) / total_frames)
            status_text.text(f"{clip_i * frames_per_clip + frame_i + 1}/{total_frames} frames processed")

    cap.release()
