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
    pos_weight: float = 1.0,
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
    pos_weight : float, optional
        The positive class weight, by default 1.
    smoothing : float, optional
        The label smoothing value, by default 0.0.
    reduction : str, optional
        The reduction method for the loss, by default "mean".
    """

    if smoothing > 0.0:
        target = target - (target - 0.5).sign() * torch.rand_like(target) * smoothing

    ce_loss = F.binary_cross_entropy_with_logits(output, target.view_as(output), reduction="none")
    p = torch.sigmoid(output)
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if pos_weight != 1:
        weight = 1 + (pos_weight - 1) * target
        loss = weight * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}. Choose from 'mean', 'sum', or 'none'.")


def multiclass_focal_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for multiclass classification.

    See CE definition: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

    Parameters
    ----------
    output : torch.Tensor
        The output tensor from the model.
    target : torch.Tensor
        The target tensor.
    gamma : float, optional
        The gamma value for the focal loss, by default 2.
    alpha : torch.Tensor | None, optional
        Class weights for each class, by default None.
    smoothing : float, optional
        The label smoothing value, by default 0.0.
    reduction : str, optional
        The reduction method for the loss, by default "mean".
    """

    if smoothing > 0.0:
        num_classes = target.size(1)
        target = (1.0 - smoothing) * target + (smoothing / num_classes)

    log_probs = F.log_softmax(output, dim=1)  # (N, C)
    probs = torch.exp(log_probs)  # (N, C)
    ce_loss = -torch.sum(target * log_probs, dim=1)  # (N,)

    p_t = torch.sum(target * probs, dim=1)  # (N,)
    focal_factor = (1.0 - p_t) ** gamma  # (N,)
    loss = focal_factor * ce_loss  # (N,)

    if alpha is not None:
        # apply class-wise weights
        class_weights = target * alpha.unsqueeze(0)  # (N, C)
        sample_weights = class_weights.sum(dim=1)  # (N,)
        loss = loss * sample_weights

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
        smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(
            inputs,
            targets,
            gamma=self.gamma,
            pos_weight=self.pos_weight,
            smoothing=self.smoothing,
            reduction=self.reduction,
        )


class MulticlassFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return multiclass_focal_loss(
            inputs,
            targets,
            gamma=self.gamma,
            alpha=self.alpha,
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
