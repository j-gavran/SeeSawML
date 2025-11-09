import glob
import logging
import os
import re
import time

import numpy as np
from omegaconf import DictConfig, open_dict

from seesaw.fakes.training.fakes_trainer import fakes_trainer
from seesaw.utils.hydra_initalize import get_hydra_config
from seesaw.utils.loggers import setup_logger


def configure_training(config: DictConfig, experiment_name: str, run_folder: str) -> tuple[DictConfig, str]:
    base_dir = f"{run_folder}/{experiment_name}"

    with open_dict(config):
        config.experiment_config.experiment_name = experiment_name
        config.experiment_config.tracker_path = f"{base_dir}/metrics/"
        config.model_config.training_config.model_save_path = f"{base_dir}/checkpoints/"

    return config, base_dir


def get_val_loss(checkpoint_path: str) -> float | None:
    match = re.search(r"val_loss=(\d+\.\d+)", checkpoint_path)

    if match:
        val_loss = float(match.group(1))
    else:
        val_loss = np.inf

    return val_loss


def configure_ratio_model(config: DictConfig, base_dir: str) -> DictConfig:
    checkpoint_dir = f"{base_dir}/checkpoints"

    num_checkpoints = glob.glob(f"{checkpoint_dir}/numModel*.ckpt")
    den_checkpoints = glob.glob(f"{checkpoint_dir}/denModel*.ckpt")

    num_losses = np.array([get_val_loss(c) for c in num_checkpoints])
    den_losses = np.array([get_val_loss(c) for c in den_checkpoints])

    if np.all(np.isinf(num_losses)) or np.all(np.isinf(den_losses)):
        logging.warning("Validation losses not available. Using first checkpoint.")
        num_idx, den_idx = 0, 0
    else:
        num_idx, den_idx = int(np.argmin(num_losses)), int(np.argmin(den_losses))

    num_checkpoint, den_checkpoint = num_checkpoints[num_idx], den_checkpoints[den_idx]

    with open_dict(config):
        config.model_config.num_config.training_config.model_save_path = f"{base_dir}/checkpoints/"
        config.model_config.den_config.training_config.model_save_path = f"{base_dir}/checkpoints/"

        config.model_config.num_config.load_checkpoint = os.path.basename(num_checkpoint)
        config.model_config.den_config.load_checkpoint = os.path.basename(den_checkpoint)

    return config


def run_training(run_folder: str, experiment_name: str | None = None, overrides: list[str] | None = None) -> None:
    if experiment_name is None:
        experiment_name = time.asctime(time.localtime()).replace("  ", "_").replace(" ", "_").replace(":", "_")

    if overrides is None:
        overrides = []

    model_names = ["num", "den", "ratio"]

    configs = {}
    for model_name in model_names:
        model_overrides = [f"model_config={model_name}"] + overrides
        configs[model_name] = get_hydra_config(
            os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"), overrides=model_overrides
        )

    for model_name in model_names:
        config, base_dir = configure_training(configs[model_name], experiment_name, run_folder)

        if model_name == "ratio":
            config = configure_ratio_model(config, base_dir)

        fakes_trainer(config)


def main() -> None:
    setup_logger()

    run_folder = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "fakes")
    os.makedirs(run_folder, exist_ok=True)

    run_training(run_folder)


if __name__ == "__main__":
    main()
