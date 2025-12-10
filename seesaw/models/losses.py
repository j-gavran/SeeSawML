import logging

import torch
import torch.nn.functional as F
from pytorch_optimizer import get_supported_loss_functions
from pytorch_optimizer.loss import LOSS_FUNCTIONS
from torch import nn


def sigmoid_focal_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for imbalanced binary classification.

    See: https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    output : torch.Tensor
        The output tensor from the model.
    target : torch.Tensor
        The target tensor.
    gamma : float, optional
        The gamma value for the focal loss, by default 2.
    smoothing : float, optional
        The label smoothing value, by default 0.0.
    reduction : str, optional
        The reduction method for the loss, by default "mean".

    Returns
    -------
    torch.Tensor
        The computed focal loss.

    References
    ----------
    - Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002

    """
    if smoothing > 0.0:
        target = target * (1 - smoothing) + 0.5 * smoothing

    ce_loss = F.binary_cross_entropy_with_logits(output, target.view_as(output), reduction="none")
    p = torch.sigmoid(output)
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Choose from 'mean', 'sum', or 'none'.")


def _smooth_labels(num_classes: int, target: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    with torch.no_grad():
        confidence = 1.0 - smoothing
        smoothing_value = smoothing / (num_classes - 1)

        # create full distribution, then fill in true class positions
        true_dist = torch.full((target.size(0), num_classes), smoothing_value, device=target.device)
        true_dist.scatter_(1, target.unsqueeze(1), confidence)

    return true_dist


def multiclass_focal_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multiclass focal loss for imbalanced multi-class classification.

    Parameters
    ----------
    output : torch.Tensor
        The output tensor from the model.
    target : torch.Tensor
        The target tensor.
    gamma : float, optional
        The gamma value for the focal loss, by default 2.0.
    smoothing : float, optional
        The label smoothing value, by default 0.0.
    reduction : str, optional
        The reduction method for the loss, by default "mean".

    Returns
    -------
    torch.Tensor
        The computed multiclass focal loss.

    References
    ----------
    - Multi-class Focal Loss: https://github.com/AdeelH/pytorch-multi-class-focal-loss

    """
    if smoothing > 0.0:
        target = _smooth_labels(output.size(1), target, smoothing)

    # compute weighted cross entropy term: -alpha * log(pt)
    # (alpha is already part of self.nll_loss)
    log_p = F.log_softmax(output, dim=-1)
    ce = F.nll_loss(log_p, target)

    # get true class column from each row
    all_rows = torch.arange(len(output))
    log_pt = log_p[all_rows, target]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt) ** gamma

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Choose from 'mean', 'sum', or 'none'.")


class FocalLoss(nn.Module):
    """Wrapper for sigmoid focal loss."""

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            inputs,
            targets,
            gamma=self.gamma,
            smoothing=self.smoothing,
            reduction=self.reduction,
        )


class MulticlassFocalLoss(nn.Module):
    """Wrapper for multiclass focal loss."""

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return multiclass_focal_loss(
            inputs,
            targets,
            gamma=self.gamma,
            smoothing=self.smoothing,
            reduction=self.reduction,
        )


def load_loss_function(loss_name: str) -> type[nn.Module]:
    if loss_name.lower() in LOSS_FUNCTIONS:
        return LOSS_FUNCTIONS[loss_name.lower()]
    else:
        raise ValueError(f"Loss function {loss_name} not supported!")


def select_nn_loss(loss_name: str, reduction="none", **kwargs) -> nn.Module:
    if loss_name == "bce" or loss_name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss(reduction=reduction, **kwargs)
    elif loss_name == "mse" or loss_name == "MSELoss":
        return torch.nn.MSELoss(reduction=reduction, **kwargs)
    elif loss_name == "ce" or loss_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(reduction=reduction, **kwargs)
    elif loss_name == "sigmoid_focal" or loss_name == "SigmoidFocalLoss":
        return FocalLoss(reduction=reduction, **kwargs)
    elif loss_name == "multiclass_focal" or loss_name == "MulticlassFocalLoss":
        return MulticlassFocalLoss(reduction=reduction, **kwargs)
    elif loss_name.lower() in get_supported_loss_functions():
        return load_loss_function(loss_name)(reduction=reduction, **kwargs)
    else:
        logging.info(f"Supported pytorch-optimizer loss functions: {get_supported_loss_functions()}")
        raise ValueError(f"Loss function {loss_name} not supported!")
