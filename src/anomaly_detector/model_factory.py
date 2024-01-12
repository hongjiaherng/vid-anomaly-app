import os
from typing import Literal

import torch
import yaml
from anomaly_detector.models import pengwu_net, sultani_net, svm_baseline

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CFG_PATHS = {
    "HL-Net": {
        "C3D": os.path.join(ROOT_PATH, "configs", "detector", "pengwu_net_c3d.yaml"),
        "I3D": os.path.join(ROOT_PATH, "configs", "detector", "pengwu_net_i3d.yaml"),
        "Video Swin": os.path.join(ROOT_PATH, "configs", "detector", "pengwu_net_swin.yaml"),
    },
    "Sultani-Net": {
        "C3D": os.path.join(ROOT_PATH, "configs", "detector", "sultani_net_c3d.yaml"),
        "I3D": os.path.join(ROOT_PATH, "configs", "detector", "sultani_net_i3d.yaml"),
        "Video Swin": os.path.join(ROOT_PATH, "configs", "detector", "sultani_net_swin.yaml"),
    },
    "SVM Baseline": {
        "C3D": os.path.join(ROOT_PATH, "configs", "detector", "svm_baseline_c3d.yaml"),
        "I3D": os.path.join(ROOT_PATH, "configs", "detector", "svm_baseline_i3d.yaml"),
        "Video Swin": os.path.join(ROOT_PATH, "configs", "detector", "svm_baseline_swin.yaml"),
    },
}
CKPT_PATHS = {
    "HL-Net": {
        "C3D": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "pengwu_net_c3d_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "pengwu_net_c3d_last.pth"),
        },
        "I3D": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "pengwu_net_i3d_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "pengwu_net_i3d_last.pth"),
        },
        "Video Swin": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "pengwu_net_swin_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "pengwu_net_swin_last.pth"),
        },
    },
    "Sultani-Net": {
        "C3D": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "sultani_net_c3d_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "sultani_net_c3d_last.pth"),
        },
        "I3D": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "sultani_net_i3d_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "sultani_net_i3d_last.pth"),
        },
        "Video Swin": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "sultani_net_swin_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "sultani_net_swin_last.pth"),
        },
    },
    "SVM Baseline": {
        "C3D": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "svm_baseline_c3d_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "svm_baseline_c3d_last.pth"),
        },
        "I3D": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "svm_baseline_i3d_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "svm_baseline_i3d_last.pth"),
        },
        "Video Swin": {
            "Best": os.path.join(ROOT_PATH, "pretrained", "detector", "svm_baseline_swin_best.pth"),
            "Last": os.path.join(ROOT_PATH, "pretrained", "detector", "svm_baseline_swin_last.pth"),
        },
    },
}


def load_config(model_name: Literal["HL-Net", "Sultani-Net", "SVM Baseline"], feature_name: Literal["C3D", "I3D", "Video Swin"]):
    cfg_path = CFG_PATHS[model_name][feature_name]

    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        raise ValueError(f"Failed to load {cfg_path=}")

    return cfg


def build_model(
    model_name: Literal["HL-Net", "Sultani-Net", "SVM Baseline"],
    feature_name: Literal["C3D", "I3D", "Video Swin"],
    ckpt_type: Literal["Best", "Last"],
):
    # Determine config path and checkpoint path
    assert model_name in CFG_PATHS, f"Unsupported model_name={model_name}"
    assert feature_name in CFG_PATHS[model_name], f"Unsupported feature_name={feature_name}"
    assert ckpt_type in ["Best", "Last"], f"Unsupported ckpt_type={ckpt_type}"

    ckpt_path = CKPT_PATHS[model_name][feature_name][ckpt_type]

    # Load config and ckpt
    model_cfg = load_config(model_name, feature_name)
    model_state_dict = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    if model_name == "HL-Net":
        feature_dim = model_cfg["dataset_cfg"]["feature_dim"]
        dropout_prob = model_cfg["model_cfg"]["dropout_prob"]
        hlc_ctx_len = model_cfg["model_cfg"]["hlc_ctx_len"]
        threshold = model_cfg["model_cfg"]["threshold"]
        sigma = model_cfg["model_cfg"]["sigma"]
        gamma = model_cfg["model_cfg"]["gamma"]

        model = pengwu_net.PengWuNet(
            feature_dim=feature_dim,
            dropout_prob=dropout_prob,
            hlc_ctx_len=hlc_ctx_len,
            threshold=threshold,
            sigma=sigma,
            gamma=gamma,
        )
        model.load_state_dict(model_state_dict)
        model.eval()

    elif model_name == "Sultani-Net":
        feature_dim = model_cfg["dataset_cfg"]["feature_dim"]
        dropout_prob = model_cfg["model_cfg"]["dropout_prob"]
        model = sultani_net.SultaniNet(feature_dim=feature_dim, dropout_prob=dropout_prob)
        model.load_state_dict(model_state_dict)
        model.eval()

    elif model_name == "SVM Baseline":
        feature_dim = model_cfg["dataset_cfg"]["feature_dim"]
        dropout_prob = model_cfg["model_cfg"]["dropout_prob"]
        model = svm_baseline.BaselineNet(feature_dim=feature_dim, dropout_prob=dropout_prob)
        model.load_state_dict(model_state_dict)
        model.eval()

    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    return model
