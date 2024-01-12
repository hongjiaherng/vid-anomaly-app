import os
import logging
from pathlib import Path

import yaml
import wandb

RUN_PATHS = {
    "pengwu_net_c3d": "jherng/wsvad/9dr5aoit",
    "pengwu_net_i3d": "jherng/wsvad/h82uyom0",
    "pengwu_net_swin": "jherng/wsvad/27seevdx",
    "sultani_net_c3d": "jherng/wsvad/agj43r4d",
    "sultani_net_i3d": "jherng/wsvad/mrqx3me5",
    "sultani_net_swin": "jherng/wsvad/f3sc03p1",
    "svm_baseline_c3d": "jherng/wsvad/9vutn2ch",
    "svm_baseline_i3d": "jherng/wsvad/fv5qg363",
    "svm_baseline_swin": "jherng/wsvad/hqx4vhgn",
}
ROOT_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))).as_posix()  # vid-anomaly-app/
CKPT_DIR = Path(f"{ROOT_PATH}/pretrained/detector").as_posix()
CFG_DIR = Path(f"{ROOT_PATH}/configs/detector").as_posix()
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_ckpt(model_name: str, wandb_api: wandb.Api):
    logger.info(f"Downloading {model_name=} from wandb (run_path={RUN_PATHS[model_name]})")
    run_data = wandb_api.run(RUN_PATHS[model_name])

    # Download ckpt
    best_ckpt_path = run_data.file(f"{run_data.name}_best.pth").download(replace=True, root=CKPT_DIR).name
    last_ckpt_path = run_data.file(f"{run_data.name}_last.pth").download(replace=True, root=CKPT_DIR).name

    # Rename ckpt
    os.rename(best_ckpt_path, f"{CKPT_DIR}/{model_name}_best.pth")
    os.rename(last_ckpt_path, f"{CKPT_DIR}/{model_name}_last.pth")

    # Retrieve config
    ckpt_config = run_data.config
    with open(f"{CFG_DIR}/{model_name}.yaml", "w") as f:
        yaml.dump(ckpt_config, f)

    return best_ckpt_path, last_ckpt_path, ckpt_config


def main():
    logger.info("Initalizing wandb's API")
    wandb_api = wandb.Api()

    logger.info(f"Downloading checkpoints to {CKPT_DIR=} and configs to {CFG_DIR=}")
    for model_name in RUN_PATHS:
        download_ckpt(model_name, wandb_api)


if __name__ == "__main__":
    main()
