import os
import logging
import requests
from pathlib import Path


RUN_PATHS = {
    "c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200": "https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth",
    "i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148": "https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth",
    "swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2": "https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth",
}
ROOT_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))).as_posix()  # vid-anomaly-app/
CKPT_DIR = Path(f"{ROOT_PATH}/pretrained/backbone").as_posix()
os.makedirs(CKPT_DIR, exist_ok=True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_ckpt(model_name: str):
    logger.info(f"Downloading {model_name=} from {RUN_PATHS[model_name]}")
    ckpt_path = Path(f"{CKPT_DIR}/{model_name}.pth").as_posix()
    r = requests.get(RUN_PATHS[model_name], allow_redirects=True)
    open(ckpt_path, "wb").write(r.content)
    return ckpt_path


def main():
    logger.info(f"Downloading checkpoints to {CKPT_DIR=}")
    for model_name in RUN_PATHS:
        download_ckpt(model_name)


if __name__ == "__main__":
    main()
