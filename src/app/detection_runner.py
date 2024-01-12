from typing import Literal, Dict, Union
import logging

import streamlit as st
import torch
from anomaly_detector.models.pengwu_net import PengWuNet
from anomaly_detector.models.sultani_net import SultaniNet
from anomaly_detector.models.svm_baseline import BaselineNet

logger = logging.getLogger(__name__)


def pengwunet_pipeline(
    clip_dict: Dict[str, torch.Tensor],
    clip_preprocessor: torch.nn.Module,
    backbone: torch.nn.Module,
    detector: PengWuNet,
    device: torch.device,
):
    # PengWuNet: (B, T=1, D) -> (B, T=1, 1)
    clip_dict = clip_preprocessor(clip_dict)  # {"inputs": (B, CROP, C, T, H, W)}; B = 1 for online inference
    clip_in = clip_dict["inputs"].squeeze(0).to(device)  # (CROP, C, T, H, W) ~ (B, C, T, H, W); CROP is effectively B here

    with torch.no_grad():
        clip_emb = backbone(clip_in)  # (B, D)
        clip_emb = clip_emb.unsqueeze(1)  # (B, 1, D) # PengWuNet expects (B, T, D) where T=1 for online inference
        clip_score = detector.predict(inputs=clip_emb, online=True)  # (B, T=1, 1)
        clip_score = torch.mean(clip_score).cpu().numpy()  # (1,)

    return clip_score


def sultaninet_baselinenet_pipeline(
    clip_dict: Dict[str, torch.Tensor],
    clip_preprocessor: torch.nn.Module,
    backbone: torch.nn.Module,
    detector: Union[SultaniNet, BaselineNet],
    device: torch.device,
):
    # SultaniNet: (B, D) -> (B, 1)
    # BaselineNet: (B, D) -> (B, 1)
    clip_dict = clip_preprocessor(clip_dict)  # {"inputs": (B, CROP, C, T, H, W)}; B = 1 for online inference
    clip_in = clip_dict["inputs"].squeeze(0).to(device)  # (CROP, C, T, H, W) ~ (B, C, T, H, W); CROP is effectively B here

    with torch.no_grad():
        clip_emb = backbone(clip_in)  # (B, D)
        clip_score = detector.predict(inputs=clip_emb)  # (B, 1)
        clip_score = torch.mean(clip_score).cpu().numpy()  # (1,)

    return clip_score


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
):
    logger.info(f"Running detection on {video_path=}")

    predict_pipeline = pengwunet_pipeline if isinstance(detector_model, PengWuNet) else sultaninet_baselinenet_pipeline
    input_video = {"path": video_path, "id": video_path}

    clip_iter = video_preprocessor(input_video)

    for clip_dict in clip_iter:
        clip_score = predict_pipeline(clip_dict, clip_preprocessor, backbone_model, detector_model, device)
        st.write(clip_score)


# def run_detection(
#     video_path: str,
#     frame_placeholder: any,
#     anomaly_score_placeholder: any,
#     frame_rate_placeholder: any,
#     resolution_placeholder: any,
#     feature_name: Literal["C3D", "I3D", "Video Swin"],
#     model_name: Literal["HL-Net", "Sultani's Net", "SVM Baseline"],
#     threshold: float,
#     enable_gpu: bool,
#     logger: logging.Logger,
# ):
#     logger.info(f"Running detection on {video_path=}")

#     input_video = init_input(video_path)
#     device = init_device(enable_gpu)

#     # Init feature extractor and pipeline
#     backbone, preprocess_video, preprocess_clip, sampling_strategy = init_backbone(feature_name, device, logger=logger)
#     # detector = init_detector(model_name, feature_name, device, logger=logger)

#     clip_iter = preprocess_video(input_video)
#     for clip_dict in clip_iter:
#         clip_dict = preprocess_clip(clip_dict)
#         clip_in = clip_dict["inputs"].squeeze(0).to(device)

#         # Run feature extraction
#         clip_embedding = backbone(clip_in)

#         st.write(clip_embedding.shape)

#         # clip_score = detector(clip_embedding)

#     # Init model

#     # Load model
#     # Loop through frames
#     # - Run prediction on frame
#     # - Put text on frame
# - Write frame, prediction result
