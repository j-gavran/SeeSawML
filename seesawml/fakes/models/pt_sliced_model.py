import logging
import os

import numpy as np
import torch
import torch.nn as nn
from f9columnar.ml.scalers import NumericalFeatureScaler
from omegaconf import DictConfig

from seesawml.fakes.models.dre_classifiers import RatioClassifier
from seesawml.models.utils import load_model_from_config, load_reports
from seesawml.utils.helpers import load_dataset_column
from seesawml.utils.hydra_initalize import get_hydra_config


class PtSlicedModelLoader:
    def __init__(
        self,
        checkpoint_path: str,
        pt_slice: tuple[float, float],
        pt_idx: int,
        column_names: list[str],
        config: DictConfig,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.pt_slice = pt_slice
        self.pt_idx = pt_idx
        self.column_names = column_names
        self.config = config

    def load_model(self) -> nn.Module:
        model, _ = load_model_from_config(self.config, RatioClassifier, checkpoint_path=self.checkpoint_path)
        return model

    def __repr__(self) -> str:
        basename = os.path.basename(self.checkpoint_path)
        model_name = ".".join(basename.split(".")[:-1])
        return f"PtSlicedModel({model_name}, pt_slice=[{self.pt_slice[0]:.2f}, {self.pt_slice[1]:.2f}] @ {self.column_names})"

    def __str__(self) -> str:
        return self.__repr__()


def get_pt_idx(column_names: list[str]) -> int:
    pt_idx, pt_idx_count = None, 0
    for i, c in enumerate(column_names):
        if "pt" in c:
            pt_idx = i
            pt_idx_count += 1

    if pt_idx_count > 1:
        raise ValueError("Multiple pt columns found in dataset")

    if pt_idx is None:
        raise ValueError("pt column not found in dataset")

    return pt_idx


def get_numer_scaler(
    config: DictConfig, numer_column_names: list[str], extra_hash: str = ""
) -> NumericalFeatureScaler | None:
    numer_scaler_type = config.dataset_config.feature_scaling.get("numer_scaler_type", None)
    scaler_path = config.dataset_config.feature_scaling.save_path

    numer_scaler = NumericalFeatureScaler(numer_scaler_type, save_path=scaler_path).load(
        column_names=numer_column_names, postfix="events_0", extra_hash=extra_hash
    )

    if numer_scaler is None:
        return None

    return numer_scaler


def scale_pt_slice(
    numer_scaler: NumericalFeatureScaler, pt_slice: tuple[float, float], pt_idx: int, numer_column_idx: np.ndarray
) -> tuple[float, float]:
    X = np.ones((2, len(numer_column_idx)), dtype=np.float32)
    X[0, pt_idx], X[1, pt_idx] = pt_slice[0], pt_slice[1]

    X_scaled = numer_scaler.transform(X)

    pt_slice_scaled = (float(X_scaled[0, pt_idx]), float(X_scaled[1, pt_idx]))

    return pt_slice_scaled


def collect_pt_sliced_models(config: DictConfig | None = None) -> list[PtSlicedModelLoader]:
    if config is None:
        config = get_hydra_config("config/ml_config", overrides=["model_config=ratio"])

    pt_sliced_model_config = config.model_config.get("pt_sliced_model", None)

    if pt_sliced_model_config is None:
        raise ValueError("pt_sliced_model config not found in model_config")

    checkpoints = pt_sliced_model_config.checkpoints

    if pt_sliced_model_config.get("model_save_path", None):
        model_save_paths = [pt_sliced_model_config.model_save_path] * len(checkpoints)
    else:
        model_save_paths = pt_sliced_model_config.model_save_paths

    events_column = load_dataset_column(
        model_save_paths[0],
        run_name=pt_sliced_model_config.checkpoints[0].split("_epoch")[0],
        dataset="events",
    )

    unsorted_column_names = set(events_column.used_columns) - set(events_column.extra_columns)
    column_names = [str(c) for c in events_column.used_columns if c in unsorted_column_names]

    numer_column_names = events_column.numer_columns
    numer_column_idx = events_column.offset_numer_columns_idx

    pt_idx = get_pt_idx(column_names)
    pt_numer_idx = get_pt_idx(numer_column_names)

    numer_scaler = get_numer_scaler(config, numer_column_names, extra_hash=config.dataset_config.files)

    pt_models = []
    for checkpoint, save_path in zip(checkpoints, model_save_paths):
        checkpoint_path = os.path.join(save_path, checkpoint)

        reports = load_reports(checkpoint_path)
        pt_slice_float = (float(reports["pt_cut"][0]), float(reports["pt_cut"][1]))

        if numer_scaler is not None:
            pt_slice_scaled = scale_pt_slice(numer_scaler, pt_slice_float, pt_numer_idx, numer_column_idx)
        else:
            pt_slice_scaled = pt_slice_float

        pt_model = PtSlicedModelLoader(checkpoint_path, pt_slice_scaled, pt_idx, column_names, config)
        pt_models.append(pt_model)

    return pt_models


class PtSlicedModel(nn.Module):
    def __init__(self, pt_models: list[PtSlicedModelLoader]) -> None:
        super().__init__()

        self.pt_models = nn.ModuleList()

        pt_slices_low, pt_slices_high = [], []
        for pt_model in pt_models:
            self.pt_models.append(pt_model.load_model())

            pt_slice = pt_model.pt_slice

            pt_slices_low.append(pt_slice[0])
            pt_slices_high.append(pt_slice[1])

        self.low_pt = nn.Parameter(torch.tensor(pt_slices_low, dtype=torch.float32), requires_grad=False)
        self.high_pt = nn.Parameter(torch.tensor(pt_slices_high, dtype=torch.float32), requires_grad=False)

        self.pt_idx = nn.Parameter(torch.tensor(pt_models[0].pt_idx, dtype=torch.int64), requires_grad=False)
        self.column_names = pt_models[0].column_names

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_pt = X[:, self.pt_idx]

        pt_splits = len(self.pt_models)
        slice_outputs = torch.zeros(size=(X.shape[0], 1), dtype=torch.float32, device=X.device)

        for i in range(pt_splits):
            low, high = self.low_pt[i], self.high_pt[i]

            if i == 0:
                mask = X_pt < high
            elif i == pt_splits - 1:
                mask = X_pt >= low
            else:
                mask = (X_pt >= low) & (X_pt < high)

            pt_model = self.pt_models[i]
            slice_outputs[mask] = pt_model(X[mask])

        return slice_outputs


def load_pt_sliced_model(config: DictConfig | None = None) -> PtSlicedModel:
    pt_models = collect_pt_sliced_models(config)
    logging.info(f"Loaded {len(pt_models)} pt-sliced models.")
    return PtSlicedModel(pt_models)


def get_dummy_sliced_model_input(pt_sliced_model: PtSlicedModel) -> torch.Tensor:
    column_names = pt_sliced_model.column_names
    low_pts, high_pts = pt_sliced_model.low_pt, pt_sliced_model.high_pt
    pt_idx = pt_sliced_model.pt_idx

    dummy_input = torch.ones(size=(len(low_pts) + len(high_pts), len(column_names)), dtype=torch.float32)

    i = 0
    for low_pt in low_pts:
        dummy_input[i, pt_idx] = low_pt.item()
        i += 1

    for high_pt in high_pts:
        dummy_input[i, pt_idx] = high_pt.item()
        i += 1

    return dummy_input
