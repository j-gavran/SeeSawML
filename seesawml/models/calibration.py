import logging
from typing import Type

import numpy as np
import torch
import torch.nn as nn


class TemperatureScaling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class VectorScaling(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_classes), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.w + self.b


class MatrixScaling(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.eye(num_classes), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits @ self.W.T + self.b


class EventsTemperatureWrapperModel(nn.Module):
    def __init__(self, model: nn.Module, temperature: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(temperature, requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.model(X)
        return logits / self.temperature


class EventsVectorScalingWrapperModel(nn.Module):
    def __init__(self, model: nn.Module, w: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.w = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.model(X)
        return logits * self.w + self.b


class EventsMatrixScalingWrapperModel(nn.Module):
    def __init__(self, model: nn.Module, W: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.W = nn.Parameter(W, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self.model(X)
        return logits @ self.W.T + self.b


class FullTemperatureWrapperModel(nn.Module):
    def __init__(self, model: nn.Module, temperature: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(temperature, requires_grad=False)

    def forward(self, X_events: torch.Tensor, *Xs: torch.Tensor) -> torch.Tensor:
        logits = self.model(X_events, *Xs)
        return logits / self.temperature


class FullVectorScalingWrapperModel(nn.Module):
    def __init__(self, model: nn.Module, w: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.w = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, X_events: torch.Tensor, *Xs: torch.Tensor) -> torch.Tensor:
        logits = self.model(X_events, *Xs)
        return logits * self.w + self.b


class FullMatrixScalingWrapperModel(nn.Module):
    def __init__(self, model: nn.Module, W: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.W = nn.Parameter(W, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, X_events: torch.Tensor, *Xs: torch.Tensor) -> torch.Tensor:
        logits = self.model(X_events, *Xs)
        return logits @ self.W.T + self.b


def get_calibration_wrapper(method: str, events_only: bool) -> Type[nn.Module]:
    if events_only:
        if method == "temperature":
            return EventsTemperatureWrapperModel
        elif method == "vector":
            return EventsVectorScalingWrapperModel
        elif method == "matrix":
            return EventsMatrixScalingWrapperModel
        else:
            raise ValueError(f"Unknown calibration method '{method}'")
    else:
        if method == "temperature":
            return FullTemperatureWrapperModel
        elif method == "vector":
            return FullVectorScalingWrapperModel
        elif method == "matrix":
            return FullMatrixScalingWrapperModel
        else:
            raise ValueError(f"Unknown calibration method '{method}'")


@torch.no_grad()
def expected_binary_calibration_error(
    probs: torch.Tensor, true_labels: torch.Tensor, weights: torch.Tensor | None = None, M: int = 15
) -> torch.Tensor:
    """Compute the expected calibration error (ECE) for binary classification.

    Parameters
    ----------
    probs : torch.Tensor
        Predicted probabilities for the positive class.
    true_labels : torch.Tensor
        True binary labels (0 or 1).
    weights : torch.Tensor | None, optional
        Sample weights, by default None.
    M : int, optional
        Number of bins to use, by default 15.

    """
    N = probs.size(0)

    # normalize weights (or uniform)
    if weights is None:
        sample_weights = torch.ones(N, dtype=probs.dtype, device=probs.device) / N
    else:
        sample_weights = weights / weights.sum()

    # bin boundaries
    bin_boundaries = torch.linspace(0, 1, M + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = true_labels.float()
    ece = torch.tensor(0.0, dtype=probs.dtype, device=probs.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)

        w_bin = sample_weights[in_bin].sum()

        if w_bin > 0:
            acc_bin = (sample_weights[in_bin] * accuracies[in_bin]).sum() / w_bin
            conf_bin = (sample_weights[in_bin] * probs[in_bin]).sum() / w_bin

            ece = ece + torch.abs(conf_bin - acc_bin) * w_bin

    return ece


@torch.no_grad()
def expected_multiclass_calibration_error(
    probs: torch.Tensor, true_labels: torch.Tensor, weights: torch.Tensor | None = None, M: int = 15
) -> torch.Tensor:
    """Compute the expected calibration error (ECE) for multiclass classification.

    Parameters
    ----------
    probs : torch.Tensor
        Predicted probabilities for each class.
    true_labels : torch.Tensor
        True class labels.
    weights : torch.Tensor | None, optional
        Sample weights, by default None.
    M : int, optional
        Number of bins to use, by default 15.

    """
    N = probs.size(0)

    # normalize weights (or create uniform weights)
    if weights is None:
        sample_weights = torch.ones(N, dtype=probs.dtype, device=probs.device) / N
    else:
        sample_weights = weights / weights.sum()

    # bin boundaries
    bin_boundaries = torch.linspace(0, 1, M + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # confidences and correctness
    confidences, predicted_labels = probs.max(dim=1)
    accuracies = (predicted_labels == true_labels).float()

    ece = torch.tensor(0.0, dtype=probs.dtype, device=probs.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        w_bin = sample_weights[in_bin].sum()

        if w_bin > 0:
            acc_bin = (sample_weights[in_bin] * accuracies[in_bin]).sum() / w_bin
            conf_bin = (sample_weights[in_bin] * confidences[in_bin]).sum() / w_bin

            ece = ece + torch.abs(conf_bin - acc_bin) * w_bin

    return ece


class Calibrator:
    def __init__(
        self,
        method: str,
        optimizer: str = "lbfgs",
        lr: float = 0.1,
        max_iter: int = 200,
        is_binary: bool = False,
    ) -> None:
        """Temperature scaling calibrator class.

        Parameters
        ----------
        method : str
            Calibration method to use. Options are 'temperature', 'vector', 'matrix'.
        optimizer : str, optional
            Optimizer to use for calibration. Options are 'lbfgs', 'lbfgs_line_search', by default 'lbfgs'.
        lr : float, optional
            Learning rate for the optimizer, by default 0.1.
        max_iter : int, optional
            Maximum number of iterations for the optimizer, by default 200.
        is_binary : bool, optional
            Whether the classification task is binary or multiclass, by default False.

        Note
        -----
        Binary classification only supports 'temperature' scaling.

        References
        ----------
        - On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
        - Probmetrics: Classification metrics and post-hoc calibration: https://github.com/dholzmueller/probmetrics
        - Temperature Scaling: https://github.com/gpleiss/temperature_scaling
        - Torch Uncertainty: https://torch-uncertainty.github.io/api.html#scaling-methods

        """

        if method not in ["temperature", "vector", "matrix"]:
            raise ValueError(f"Unknown calibration method '{method}'.")

        if optimizer not in ["lbfgs", "lbfgs_line_search"]:
            raise ValueError(f"Unknown optimizer '{optimizer}' for calibration.")

        if is_binary and method != "temperature":
            raise ValueError("Only 'temperature' scaling is supported for binary classification.")

        self.method = method

        self.opt = optimizer
        self.lr = lr
        self.max_iter = max_iter

        self.is_binary = is_binary

        self._is_fitted = False

    def _raise_fit_error(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Calibrator has not been fitted yet. Call 'fit' method first!")

    def _fit_lbfgs(self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> None:
        self.param: nn.Module

        num_classes = logits.shape[1]

        if self.method == "temperature":
            self.param = TemperatureScaling()
        elif self.method == "vector":
            self.param = VectorScaling(num_classes)
        elif self.method == "matrix":
            self.param = MatrixScaling(num_classes)
        else:
            raise ValueError(f"Unknown scaling method '{self.method}'")

        self.param = self.param.to(logits.device)

        criterion: nn.Module
        if self.is_binary:
            criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(reduction="none")

        optimizer = torch.optim.LBFGS(
            self.param.parameters(),
            lr=self.lr,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe" if self.opt == "lbfgs_line_search" else None,
        )

        if weights is None:
            weights = torch.ones_like(labels, dtype=torch.float32)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            y_pred = self.param(logits)
            loss = torch.mean(criterion(y_pred, labels) * weights)
            loss.backward()
            return loss

        optimizer.step(closure)

        self._is_fitted = True
        self._calculate_ece(logits, labels, weights)

    def _calculate_ece(self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> None:
        self._raise_fit_error()

        if self.is_binary:
            probs_before = torch.sigmoid(logits.squeeze())
            probs_after = torch.sigmoid(self.param(logits).squeeze())

            targets = labels.to(logits.dtype)

            ece_before = expected_binary_calibration_error(probs_before, targets, weights)
            ece_after = expected_binary_calibration_error(probs_after, targets, weights)
        else:
            probs_before = torch.softmax(logits, dim=-1)
            probs_after = torch.softmax(self.param(logits), dim=-1)

            targets = labels.to(torch.int64)

            ece_before = expected_multiclass_calibration_error(probs_before, targets, weights)
            ece_after = expected_multiclass_calibration_error(probs_after, targets, weights)

        self._ece_score = (ece_before.item(), ece_after.item())

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> None:
        self._fit_lbfgs(logits, labels, weights)

    def params(self) -> dict[str, torch.Tensor]:
        self._raise_fit_error()

        params: dict[str, torch.Tensor]

        params = {k: v.detach() for k, v in self.param.state_dict().items()}

        return params

    def ece_score(self) -> tuple[float, float]:
        self._raise_fit_error()

        return self._ece_score


def get_calibration_split(split_dct: dict[str, int], add_calib_train: bool = False) -> dict[str, list[int]]:
    if "calib" not in split_dct:
        raise ValueError("No 'calib' key found in split_dct for calibration split!")

    new_split_dct = {}
    start, stop = 0, 0
    for key, value in split_dct.items():
        stop += value
        new_split_dct[key] = np.arange(start, stop, 1).tolist()
        start += value

    if add_calib_train:
        calib_len = len(new_split_dct["calib"])
        calib_start = new_split_dct["calib"][0]
        new_split_dct["train"] = np.arange(calib_start, calib_start + calib_len, 1).tolist()
        new_split_dct.pop("calib")

    return new_split_dct
