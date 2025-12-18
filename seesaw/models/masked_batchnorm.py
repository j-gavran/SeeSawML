import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def masked_batch_norm(
    input: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    momentum: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Applies Masked Batch Normalization for each channel in each data sample in a batch.

    Source: https://gist.github.com/ilya16/c622461000480e66ae906dd9dbe8ea26

    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError("Expected running_mean and running_var to be not None when training=False")

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements  # (C,)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements  # (C,)

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var  # type: ignore[assignment]

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_MaskedBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        # passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        # used for normalization (i.e. in eval mode when buffers are not None).
        if mask is None:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            return masked_batch_norm(
                input,
                mask,
                self.weight,
                self.bias,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                bn_training,
                exponential_average_factor,
                self.eps,
            )


class MaskedBatchNorm1d(nn.BatchNorm1d, _MaskedBatchNorm):
    """Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super(MaskedBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)


class MaskedBatchNorm2d(nn.BatchNorm2d, _MaskedBatchNorm):
    """Applies Batch Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension)..

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super(MaskedBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)


class MaskedBatchNorm3d(nn.BatchNorm3d, _MaskedBatchNorm):
    """Applies Batch Normalization over a masked 5D input
    (a mini-batch of 3D inputs with additional channel dimension).

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Mask: :math:`(N, 1, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super(MaskedBatchNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
