import importlib
import logging
from typing import Dict, Literal, Optional, Tuple, Union

import anomaly_detector.model_factory as model_factory
import streamlit as st
import torch
from anomaly_detector.models.pengwu_net import PengWuNet
from anomaly_detector.models.sultani_net import SultaniNet
from anomaly_detector.models.svm_baseline import BaselineNet

logger = logging.getLogger(__name__)


def load_backbone(
    feature_name: Literal["C3D", "I3D", "Video Swin"],
    device: torch.device,
    crop_type: Literal["10-crop", "5-crop", "center"] = "5-crop",
) -> Tuple[
    Optional[torch.nn.Module],
    Optional[torch.nn.Module],
    Optional[torch.nn.Module],
    Optional[Dict[str, int]],
]:
    @st.cache_resource(show_spinner="Loading backbone model...")
    def _load_backbone(
        feature_name: Literal["C3D", "I3D", "Video Swin"],
        device: torch.device,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Dict[str, int]]:
        if feature_name == "C3D":
            backbone_module = importlib.import_module("feature_extractor.models.c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb")
        elif feature_name == "I3D":
            backbone_module = importlib.import_module(
                "feature_extractor.models.i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb"
            )
        elif feature_name == "Video Swin":
            backbone_module = importlib.import_module("feature_extractor.models.swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb")
        else:
            raise ValueError(f"Unsupported feature_name={feature_name}")

        backbone = backbone_module.build_model().to(device)
        preprocess_video = backbone_module.build_video2clips_pipeline(batch_size=1, io_backend="local", id_key="id", path_key="path", num_clips=-1)
        preprocess_clip = backbone_module.build_clip_pipeline(crop_type=crop_type)
        sampling_strategy = {
            "clip_len": backbone_module.preprocessing_cfg["clip_len"],
            "sampling_rate": backbone_module.preprocessing_cfg["sampling_rate"],
        }
        logger.info(f"Initialized {feature_name} backbone - {device}")

        return backbone, preprocess_video, preprocess_clip, sampling_strategy

    try:
        return _load_backbone(feature_name, device)

    except Exception as e:
        logger.error(e)
        st.error(f"Failed to load {feature_name} backbone")
        return None, None, None, None


def load_detector(
    model_name: Literal["HL-Net", "Sultani-Net", "SVM Baseline"],
    feature_name: Literal["C3D", "I3D", "Video Swin"],
    ckpt_type: Literal["Best", "Last"],
    device: torch.device,
) -> Optional[Union[PengWuNet, SultaniNet, BaselineNet]]:
    @st.cache_resource(show_spinner="Loading detector model...")
    def _load_detector(
        model_name: Literal["HL-Net", "Sultani-Net", "SVM Baseline"],
        feature_name: Literal["C3D", "I3D", "Video Swin"],
        ckpt_type: Literal["Best", "Last"],
        device: torch.device,
    ) -> Union[PengWuNet, SultaniNet, BaselineNet]:
        model = model_factory.build_model(model_name, feature_name, ckpt_type).to(device)
        logger.info(f"Initialized {model_name} ({feature_name=}, {ckpt_type=}) detector - {device}")

        return model

    try:
        return _load_detector(model_name, feature_name, ckpt_type, device)
    except Exception as e:
        logger.error(e)
        st.error(f"Failed to load {model_name} ({feature_name=}, {ckpt_type=}) detector - {device}")
        return None
