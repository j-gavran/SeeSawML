import math
from typing import Any

import torch
from einops import rearrange
from torch import nn

from seesaw.models.activations import get_activation
from seesaw.models.transformers.attention import CrossAttention


class FeedForwardLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        act: str | None,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        pre_act: bool = False,
        act_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}

        self.layers = nn.ModuleList()

        if pre_act:
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(n_in))

            self.layers.append(get_activation(act, **act_kwargs))
            self.layers.append(nn.Linear(n_in, n_out))

            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))
        else:
            self.layers.append(nn.Linear(n_in, n_out))

            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))

            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(n_in))

            self.layers.append(get_activation(act, **act_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualFeedForwardLayer(nn.Module):
    def __init__(
        self,
        layers_dim: list[int],
        act: str | None,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        **act_kwargs: Any,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layers_dim) - 1):
            _n_in, _n_out = layers_dim[i], layers_dim[i + 1]

            layer = FeedForwardLayer(
                n_in=_n_in,
                n_out=_n_out,
                act=act,
                use_batchnorm=use_batchnorm,
                dropout=0.0,
                pre_act=True,
                **act_kwargs,
            )
            self.layers.add_module(f"layer{i}", layer)

            if i < len(layers_dim) - 2 and dropout > 0.0:
                self.layers.add_module(f"dropout{i}", nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for layer in self.layers:
            x = layer(x)
        return x + z


class FeatureWiseLinear(nn.Module):
    def __init__(self, n_features: int, n_out: int, feature_idx: int = 1, bias: bool = True) -> None:
        """Layer that applies a separate linear transformation to each feature independently.

        Parameters
        ----------
        n_features : int
            Number of features to transform.
        n_out : int
            Output dimension for each feature.
        feature_idx : int, optional
            Index of the feature dimension in the input tensor, by default 1.
        bias : bool, optional
            Whether to include a bias term in the linear transformations, by default True.
        """
        super().__init__()
        self.feature_idx = feature_idx
        self.n_features = n_features

        self.linear_layers = nn.ModuleList([nn.Linear(1, n_out, bias=bias) for _ in range(n_features)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_split = torch.unbind(x, dim=self.feature_idx)

        x_lin = [self.linear_layers[i](x_split[i].unsqueeze(-1)) for i in range(self.n_features)]

        return torch.stack(x_lin, dim=self.feature_idx)


class StackedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, channels: int) -> None:
        """Efficient implementation of linear layers for ensembles of networks.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        channels : int
            Number of linear layers (channels) to stack.

        References
        ----------
        [1] - https://github.com/luigifvr/ljubljana_ml4physics_25/blob/main/src/stackedlinear.py

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels

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
        return torch.baddbmm(self.bias[:, None, :], x, self.weight.transpose(1, 2))


class MeanPoolingLayer(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False) -> None:
        """Mean pooling layer.

        Parameters
        ----------
        dim : int | None, optional
            Dimension along which to compute the mean, by default None.
        keepdim : bool, optional
            Whether to keep the dimension after reduction, by default False.

        Note
        ----
        If a mask is provided during the forward pass, the mean is computed only over the valid elements.

        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

        x = x.masked_fill(mask, 0.0)
        valid = (~mask).sum(dim=self.dim, keepdim=True).clamp(min=1)
        x = x.sum(dim=self.dim, keepdim=True) / valid

        return x.squeeze(self.dim) if not self.keepdim else x


class FlatJaggedModelFuser(nn.Module):
    def __init__(self, fuse_mode: str, output_dim: int | None, fuse_kwargs: dict[str, Any] | None = None) -> None:
        """Fusing layer to combine flat and jagged inputs coming from a flat model and a jagged model.

        Parameters
        ----------
        fuse_mode : str
            Mode of fusion. Options are 'add', 'cat' (recommended), 'learn', 'gate', 'attn' (experimental).
        output_dim : int | None
            Expected output dimension. If None, will be 1.
        fuse_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments for specific fusion modes, by default None.

        Note
        ----
        Should be used at the end of a model combining flat and jagged inputs.

        """
        super().__init__()
        if output_dim is None:
            output_dim = 1

        if fuse_kwargs is None:
            fuse_kwargs = {}

        self.fuse_add = fuse_mode == "add"
        self.fuse_cat = fuse_mode == "cat"
        self.fuse_learn = fuse_mode == "learn"
        self.fuse_gate = fuse_mode == "gate"
        self.fuse_attn = fuse_mode == "attn"

        if self.fuse_cat:
            self.cat_model = nn.Linear(2 * output_dim, output_dim)

        if self.fuse_learn:
            self.alpha_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        if self.fuse_gate:
            self.gate = nn.Sequential(nn.Linear(2 * output_dim, output_dim), nn.Sigmoid())

        if self.fuse_attn:
            embedding_dim = fuse_kwargs.get("embedding_dim", 64)

            self.flat_proj = nn.Linear(output_dim, embedding_dim)
            self.jagged_proj = nn.Linear(output_dim, embedding_dim)

            self.attn_layer = CrossAttention(
                dim=embedding_dim,
                heads=fuse_kwargs.get("heads", 4),
                dim_head=fuse_kwargs.get("dim_head", 16),
                attn_dropout=fuse_kwargs.get("dropout", 0.1),
                normalize_q=True,
                dim_out=output_dim,
                sdp_backend=fuse_kwargs.get("sdp_backend", None),
            )

    def forward(self, flat_x: torch.Tensor, jagged_x: torch.Tensor) -> torch.Tensor:
        if self.fuse_add:
            return flat_x + jagged_x
        elif self.fuse_cat:
            return self.cat_model(torch.cat([flat_x, jagged_x], dim=-1))
        elif self.fuse_learn:
            alpha = torch.sigmoid(self.alpha_param)
            return alpha * flat_x + (1 - alpha) * jagged_x
        elif self.fuse_gate:
            gate_score = self.gate(torch.cat([flat_x, jagged_x], dim=-1))
            return gate_score * flat_x + (1 - gate_score) * jagged_x
        elif self.fuse_attn:
            flat_x_proj = self.flat_proj(flat_x).unsqueeze(1)
            jagged_x_proj = self.jagged_proj(jagged_x).unsqueeze(1)

            attn = self.attn_layer(jagged_x_proj, flat_x_proj)
            return attn.squeeze(1)
        else:
            raise RuntimeError(f"Fuse mode {self.fuse_add} not recognized!")


class FlatJaggedEmbeddingsFuser(nn.Module):
    def __init__(
        self,
        fuse_mode: str,
        output_dim: int,
        fuse_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Fusing layer to combine flat and jagged inputs coming from flat embeddings and jagged embeddings.

        Parameters
        ----------
        fuse_mode : str
            Mode of fusion. Options are 'sum', 'mean', 'add', 'cat' (recommended), 'cat_proj', 'learn', 'gate',
            'attn' (experimental), 'res_attn' (experimental).
        output_dim : int
            Expected output dimension.
        fuse_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments for specific fusion modes, by default None.

        Note
        ----
        Should be used at the end of a model combining flat and jagged inputs.

        """
        super().__init__()

        if fuse_kwargs is None:
            fuse_kwargs = {}

        self.fuse_sum = fuse_mode == "sum"
        self.fuse_mean = fuse_mode == "mean"
        self.fuse_add = fuse_mode == "add"
        self.fuse_cat = fuse_mode == "cat"
        self.fuse_cat_proj = fuse_mode == "cat_proj"
        self.fuse_learn = fuse_mode == "learn"
        self.fuse_gate = fuse_mode == "gate"
        self.fuse_attn = fuse_mode == "attn"
        self.fuse_res_attn = fuse_mode == "res_attn"

        if self.fuse_cat_proj:
            self.cat_project = nn.Linear(2 * output_dim, output_dim)

        if self.fuse_learn:
            self.alpha_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        if self.fuse_gate:
            self.gate = nn.Sequential(nn.Linear(2 * output_dim, output_dim), nn.Sigmoid())

        if self.fuse_attn:
            self.attn_layer = CrossAttention(
                dim=output_dim,
                heads=fuse_kwargs.get("heads", 4),
                dim_head=fuse_kwargs.get("dim_head", 16),
                attn_dropout=fuse_kwargs.get("dropout", 0.1),
                normalize_q=True,
                dim_out=output_dim,
                sdp_backend=fuse_kwargs.get("sdp_backend", None),
            )

    def forward(self, flat_x: torch.Tensor, jagged_x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.fuse_attn:
            flat_x = rearrange(flat_x, "b e -> b 1 e")

        if self.fuse_sum:
            fused = jagged_x + torch.sum(flat_x, dim=-1, keepdim=True)
        elif self.fuse_mean:
            fused = jagged_x + torch.mean(flat_x, dim=-1, keepdim=True)
        elif self.fuse_add:
            fused = flat_x + jagged_x
        elif self.fuse_cat:
            flat_x_expanded = flat_x.expand(-1, jagged_x.shape[1], -1)
            fused = torch.cat([flat_x_expanded, jagged_x], dim=-1)
        elif self.fuse_cat_proj:
            flat_x_expanded = flat_x.expand(-1, jagged_x.shape[1], -1)
            fused = self.cat_project(torch.cat([flat_x_expanded, jagged_x], dim=-1))
        elif self.fuse_learn:
            alpha = torch.sigmoid(self.alpha_param)
            fused = alpha * flat_x + (1 - alpha) * jagged_x
        elif self.fuse_gate:
            flat_x_expanded = flat_x.expand(-1, jagged_x.shape[1], -1)
            gate_score = self.gate(torch.cat([flat_x_expanded, jagged_x], dim=-1))
            fused = gate_score * flat_x + (1 - gate_score) * jagged_x
        elif self.fuse_attn:
            fused = self.attn_layer(jagged_x, context=flat_x)
        elif self.fuse_res_attn:
            attn_out = self.attn_layer(jagged_x, context=flat_x)
            fused = jagged_x + attn_out
        else:
            raise RuntimeError("Fuse mode not recognized!")

        if mask is not None:
            mask = rearrange(mask, "b o -> b o 1")
            fused = fused.masked_fill(mask, 0.0)

        return fused
