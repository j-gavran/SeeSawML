import torch
from torch import nn

from seesaw.models.activations import GeGLU, SwiGLU


class ReLUNet(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        dim_out: int | None = None,
        output_dropout: bool = True,
    ) -> None:
        super().__init__()
        if mult <= 0:
            raise ValueError("mult must be a positive integer!")

        if dim_out is None:
            dim_out = dim

        layers: list[nn.Module] = []

        if use_layernorm:
            layers += [nn.LayerNorm(dim)]

        if dropout > 0.0:
            layers += [
                nn.Linear(dim, dim * mult),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mult, dim_out),
                nn.Dropout(dropout),
            ]
            if not output_dropout:
                layers = layers[:-1]
        else:
            layers += [
                nn.Linear(dim, dim * mult),
                nn.ReLU(),
                nn.Linear(dim * mult, dim_out),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GELUNet(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        dim_out: int | None = None,
        output_dropout: bool = True,
    ) -> None:
        super().__init__()
        if dim_out is None:
            dim_out = dim

        layers: list[nn.Module] = []

        if use_layernorm:
            layers += [nn.LayerNorm(dim)]

        if dropout > 0.0:
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim_out),
                nn.Dropout(dropout),
            ]
            if not output_dropout:
                layers = layers[:-1]
        else:
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim_out),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GeGLUNet(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        dim_out: int | None = None,
        output_dropout: bool = True,
    ) -> None:
        super().__init__()
        if mult <= 0:
            raise ValueError("mult must be a positive integer!")

        if dim_out is None:
            dim_out = dim

        layers: list[nn.Module] = []

        if use_layernorm:
            layers += [nn.LayerNorm(dim)]

        if dropout > 0.0:
            layers += [
                nn.Linear(dim, dim * mult * 2),
                GeGLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mult, dim_out),
                nn.Dropout(dropout),
            ]
            if not output_dropout:
                layers = layers[:-1]
        else:
            layers += [
                nn.Linear(dim, dim * mult * 2),
                GeGLU(),
                nn.Linear(dim * mult, dim_out),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SwiGLUNet(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        dim_out: int | None = None,
        output_dropout: bool = True,
    ) -> None:
        super().__init__()
        if mult <= 0:
            raise ValueError("mult must be a positive integer!")

        if dim_out is None:
            dim_out = dim

        layers: list[nn.Module] = []

        if use_layernorm:
            layers += [nn.LayerNorm(dim)]

        if dropout > 0.0:
            layers += [
                nn.Linear(dim, dim * mult * 2),
                SwiGLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mult, dim_out),
                nn.Dropout(dropout),
            ]
            if not output_dropout:
                layers = layers[:-1]
        else:
            layers += [
                nn.Linear(dim, dim * mult * 2),
                SwiGLU(),
                nn.Linear(dim * mult, dim_out),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
