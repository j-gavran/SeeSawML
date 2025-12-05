from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig


class Tracker(ABC):
    def __init__(self, experiment_conf: DictConfig, tracker_path: str) -> None:
        self.experiment_conf = experiment_conf
        self.tracker_path = tracker_path

        self.check_metrics_n_epoch = experiment_conf.get("check_metrics_n_epoch", None)
        self.plot_metrics_n_epoch = experiment_conf.get("plot_metrics_n_epoch", None)

        self.skip_check_metrics = True if self.check_metrics_n_epoch is None else False
        self.skip_plot_metrics = True if self.plot_metrics_n_epoch is None else False

        if self.skip_check_metrics and not self.skip_plot_metrics:
            raise ValueError("Cannot skip metric checks while plotting metrics. Set both to None or specify epochs.")

        if self.skip_check_metrics:
            logging.info("Skipping metric checks.")
        else:
            logging.info(f"Tracking metrics every: {self.check_metrics_n_epoch} epochs.")

        if self.skip_plot_metrics:
            logging.info("Skipping metric plotting.")
        else:
            logging.info(f"Plotting metrics every: {self.plot_metrics_n_epoch} epochs.")

        self.module: L.LightningModule
        self.current_epoch: int
        self.stage: str

        self.base_dir = f"{self.tracker_path}/{self.experiment_conf['run_name']}/"

        self._create_plotting_dirs()

        self._class_weights: dict[str, dict[int, float]] | None = None

    def __call__(self, module: L.LightningModule) -> Tracker:
        self.module = module
        return self

    def _create_plotting_dirs(self) -> None:
        for d in list(self.plotting_dirs.values()):
            os.makedirs(d, exist_ok=True)

    @property
    @abstractmethod
    def plotting_dirs(self) -> dict[str, str]:
        pass

    def validate_compute(self) -> bool:
        if self.stage == "test":
            return True
        elif self.skip_check_metrics:
            return False
        elif self.current_epoch % self.check_metrics_n_epoch != 0:
            return False
        else:
            return True

    def validate_plot(self) -> bool:
        if self.stage == "test":
            return True
        elif self.skip_plot_metrics:
            return False
        elif self.current_epoch % self.plot_metrics_n_epoch != 0:
            return False
        else:
            return True

    def log_artifacts(self) -> None:
        try:
            local_path, run_id = self.base_dir, self.module.logger.run_id  # type: ignore
            self.module.logger.experiment.log_artifact(local_path=local_path, run_id=run_id)  # type: ignore
        except AttributeError:
            pass

    @abstractmethod
    def compute(self, batch: Any, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if not self.validate_compute():
            return False

        return True

    @abstractmethod
    def plot(self, stage: str) -> bool:
        if not self.validate_plot():
            return False

        self.log_artifacts()

        return True

    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker state."""
        # Reset any accumulated state here if necessary
        # For example, if you have lists or tensors to accumulate results, clear them.
        # self.accumulated_results.clear()  # Example placeholder
        pass

    def to_int64_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.to(torch.int64).cpu().numpy()

    def to_float32_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.to(torch.float32).cpu().numpy()

    def add_class_weights(self, class_weights: dict[str, dict[int, float]]) -> None:
        self._class_weights = class_weights

    @property
    def has_class_weights(self) -> bool:
        return True if self._class_weights is not None else False

    @property
    def class_weights_dct(self) -> dict[str, dict[int, float]]:
        if self._class_weights is None:
            raise ValueError("Class weights have not been set. Please use 'add_class_weights' to set them.")
        return self._class_weights

    def get_class_weights_tensor(
        self, stage: str, norm: bool = False, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        class_weights = self.class_weights_dct[stage]
        weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=dtype)

        if norm:
            weights_tensor = weights_tensor / torch.sum(weights_tensor)

        return weights_tensor

    def get_class_weights_labels(
        self, labels: torch.Tensor, stage: str, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        _class_weights = self.class_weights_dct[stage]

        class_weights = torch.zeros_like(labels, dtype=dtype)

        for class_label, class_weight in _class_weights.items():
            class_weights[labels == class_label] = class_weight

        return class_weights
