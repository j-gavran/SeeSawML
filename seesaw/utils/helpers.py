import logging
import multiprocessing
import os

import numpy as np
import torch
from f9columnar.ml.dataloader_helpers import DatasetColumn, column_selection_from_dict
from f9columnar.utils.helpers import load_json
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import train_test_split


def get_splits(
    n_data: int,
    train_split: float,
    val_split: float | None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list] | tuple[np.ndarray, list, list]:
    idx = np.arange(n_data)

    if train_split == 1.0:
        if shuffle:
            np.random.shuffle(idx)
        return idx, [], []

    remaining, train_idx = train_test_split(idx, test_size=train_split, shuffle=shuffle)

    if val_split is None:
        return train_idx, remaining, []
    else:
        test_idx, val_idx = train_test_split(idx[remaining], test_size=val_split, shuffle=shuffle)

    return train_idx, val_idx, test_idx


def get_log_binning(x_min: float, x_max: float, nbins: int) -> np.ndarray:
    return np.logspace(np.log(x_min), np.log(x_max), nbins, base=np.e)


def bin_edges_from_centers(bin_centers: np.ndarray) -> np.ndarray:
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.concatenate(([bin_centers[0] - bin_width / 2], bin_centers + bin_width / 2))
    return bin_edges


def to_cpu_numpy(*tensors: torch.Tensor) -> list[np.ndarray]:
    cpu_numpy_tensors = []
    for tensor in tensors:
        cpu_numpy_tensors.append(tensor.cpu().numpy())

    return cpu_numpy_tensors


def setup_analysis_dirs(config: DictConfig, verbose: bool = True) -> None:
    with open_dict(config):
        if config.experiment_config.get("save_dir", None) is None:
            config.experiment_config.save_dir = os.path.join(os.environ["ANALYSIS_ML_LOGS_DIR"], "mlruns")

        if config.experiment_config.get("tracker_path", None) is None:
            config.experiment_config.tracker_path = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "metrics")

        if config.model_config.training_config.get("model_save_path", None) is None:
            config.model_config.training_config.model_save_path = os.path.join(
                os.environ["ANALYSIS_ML_MODELS_DIR"], "checkpoints"
            )

        if config.dataset_config.feature_scaling.get("save_path", None) is None:
            config.dataset_config.feature_scaling.save_path = os.path.join(
                os.environ["ANALYSIS_ML_RESULTS_DIR"], "scalers"
            )

        config.dataset_config.files = os.path.join(os.environ["ANALYSIS_ML_DATA_DIR"], config.dataset_config.files)

    if verbose:
        logging.info(f"Saving mlflow logs in {config.experiment_config.save_dir}.")
        logging.info(f"Saving tracker metrics in {config.experiment_config.tracker_path}.")
        logging.info(f"Checkpointing models in {config.model_config.training_config.model_save_path}.")
        logging.info(f"Using hdf5 data from {config.dataset_config.files}.")
        logging.info(f"Using scalers from {config.dataset_config.feature_scaling.save_path}.")


def load_dataset_column(model_save_path: str, run_name: str, dataset: str) -> DatasetColumn:
    selection_path = os.path.join(model_save_path, f"{run_name}_selection.json")
    selection_dct = load_json(selection_path)

    selection = column_selection_from_dict(selection_dct)

    return selection[dataset]


def load_dataset_column_from_config(config: DictConfig, dataset: str) -> DatasetColumn:
    return load_dataset_column(
        config.model_config.training_config.model_save_path,
        config.model_config.load_checkpoint.split("_epoch")[0],
        dataset,
    )


def verify_num_workers(num_workers: int, stage_split_piles: dict[str, int | list[int]]) -> None:
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    for key, values in stage_split_piles.items():
        if isinstance(values, list):
            value = len(values)
        else:
            value = values

        if num_workers > value and value != 0:
            raise ValueError(
                f"Number of workers {num_workers} cannot be greater than the number of {key} samples {value}!"
            )


class InvalidConfigError(Exception):
    def __init__(self, message: str = "Invalid configuration!", errors: dict | None = None):
        super().__init__(message)
        self.errors = errors

    def __str__(self):
        base_message = super().__str__()

        if self.errors:
            detailed_errors = "\n".join(f"{key}: {value}" for key, value in self.errors.items())
            return f"{base_message}\nDetails:\n{detailed_errors}"

        return base_message
