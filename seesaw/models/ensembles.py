import math

import numpy as np
import torch
from torch import nn

from seesaw.models.activations import get_activation


class StackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, channels: int, expand_first: bool = False) -> None:
        """Efficient implementation of linear layers for ensembles of networks.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        channels : int
            Number of linear layers (channels) to stack.
        expand_first : bool, optional
            Whether to expand the input tensor along the first dimension, by default False.

        References
        ----------
        [1] - https://github.com/luigifvr/ljubljana_ml4physics_25/blob/main/src/stackedlinear.py

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.expand_first = expand_first

        self.weight = nn.Parameter(torch.empty((channels, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((channels, out_features)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.channels):
            torch.nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand_first:
            x = x.unsqueeze(0).expand(self.channels, -1, -1)

        ensemble = torch.baddbmm(self.bias[:, None, :], x, self.weight.transpose(1, 2))

        return ensemble  # (channels, batch, out)


class StackedLinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channels: int,
        act: str | None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.linear = StackedLinear(in_features, out_features, channels)
        self.act = get_activation(act)

        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.use_layernorm = use_layernorm

        if self.use_layernorm:
            self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.act(x)

        if self.use_dropout:
            x = self.dropout(x)

        if self.use_layernorm:
            x = self.norm(x)

        return x


class StackedEnsembleNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        act: str | None = None,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        expand_first: bool = True,
        use_log_var: bool = True,
    ) -> None:
        """A neural network for regression with a Gaussian likelihood.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        channels : int
            Number of ensemble members (channels).
        act : str | None, optional
            Activation function to use, by default None (no activation).
        hidden_dims : list[int] | None, optional
            List of hidden layer dimensions, by default None (no hidden layers).
        output_dim : int, optional
            Number of output features, by default 1.
        dropout : float, optional
            Dropout rate to use, by default 0.0 (no dropout).
        use_layernorm : bool, optional
            Whether to use layer normalization, by default False.
        expand_first : bool, optional
            Whether to expand the input tensor along the first dimension, by default True.
        use_log_var : bool, optional
            Whether to output log variance, by default True. If False, the model will mean only.
            Set to False if using this model for a BCE ensemble, where we only
            care about the mean/logits.
        """
        super().__init__()
        self.channels = channels
        self.expand_first = expand_first

        hidden_dims = [] if hidden_dims is None else hidden_dims

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(StackedLinearBlock(prev_dim, hidden_dim, channels, act, dropout, use_layernorm))
            prev_dim = hidden_dim

        self.net: nn.Module

        if len(layers) > 0:
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Identity()

        if use_log_var:
            mult = 2
        else:
            mult = 1

        self.mean_log_var_layer = StackedLinear(prev_dim, mult * output_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand_first:
            x = x.unsqueeze(0).expand(self.channels, -1, -1)

        x = self.net(x)

        return self.mean_log_var_layer(x)  # (channels, batch, 2 * output_dim) or (channels, batch, output_dim)


class StackedEnsembleNetWrapper(nn.Module):
    def __init__(
        self,
        backbone_model: torch.nn.Module,
        backbone_output_dim: int,
        channels: int = 1,
        act: str | None = None,
        act_out: str | None = None,
        hidden_dims: list[int] | None = None,
        ensemble_output_dim: int = 1,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        use_log_var: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_model = backbone_model

        self.ensemble_net = StackedEnsembleNet(
            input_dim=backbone_output_dim,
            channels=channels,
            act=act,
            hidden_dims=hidden_dims,
            output_dim=ensemble_output_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
            expand_first=True,
            use_log_var=use_log_var,
        )

        self.act_out = get_activation(act_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone_model(x)
        return self.act_out(self.ensemble_net(x))


@torch.no_grad()
def torch_predict_from_ensemble_gauss_nll(
    model_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    re_mean_pred, re_log_var_pred = model_output.chunk(2, dim=-1)

    re_mean_pred = re_mean_pred.squeeze(-1)
    re_var_pred = torch.exp(re_log_var_pred).squeeze(-1)
    # calculate the overall mean and variance of the predictions
    re_mean = torch.mean(re_mean_pred, dim=0)

    # tends to be large where the data is inherently noisy (regardless of how much data you have)
    # how noisy does the model think the data is at this location
    # also called "aleatoric uncertainty"
    re_variance_syst = torch.mean(re_var_pred, dim=0)

    # tends to be large where you have little data (the model is uncertain)
    # how confident/consistent is the model about its prediction
    # also called "epistemic uncertainty"
    re_variance_stat = torch.var(re_mean_pred, dim=0)

    re_std_syst = torch.sqrt(re_variance_syst)
    re_std_stat = torch.sqrt(re_variance_stat)

    re_variance_total = re_variance_syst + re_variance_stat
    re_std_total = torch.sqrt(re_variance_total)

    return re_mean, re_std_syst, re_std_stat, re_std_total


@torch.no_grad()
def torch_predict_from_ensemble_logits(model_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensemble prediction in logit (log-ratio) space for BCE models.

    Returns only the epistemic uncertainty (ensemble disagreement of logits).

    """

    logits = model_output.squeeze(-1)  # (channels, batch)

    re_mean = torch.mean(logits, dim=0)
    re_std = torch.std(logits, dim=0)

    return re_mean, re_std


def np_predict_from_ensemble_logits(model_output: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble prediction in logit (log-ratio) space for BCE models.

    Returns only the epistemic uncertainty (ensemble disagreement of logits).

    """
    logits = model_output.squeeze(-1).numpy()  # (channels, batch)

    re_mean = np.mean(logits, axis=0)
    re_std = np.std(logits, axis=0)

    return re_mean, re_std


def np_predict_from_ensemble_gauss_nll(
    model_output: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    re_mean_pred, re_log_var_pred = model_output.chunk(2, dim=-1)

    re_mean_pred = re_mean_pred.squeeze(-1).numpy()  # type: ignore[assignment]
    re_var_pred = torch.exp(re_log_var_pred).squeeze(-1).numpy()

    re_mean = np.mean(re_mean_pred, axis=0)

    re_variance_syst = np.mean(re_var_pred, axis=0)
    re_variance_stat = np.var(re_mean_pred, axis=0)

    re_std_syst = np.sqrt(re_variance_syst)
    re_std_stat = np.sqrt(re_variance_stat)

    re_variance_total = re_variance_syst + re_variance_stat
    re_std_total = np.sqrt(re_variance_total)

    return re_mean, re_std_syst, re_std_stat, re_std_total
