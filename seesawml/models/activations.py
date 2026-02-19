import torch
import torch.nn as nn


class GeGLU(nn.Module):
    def __init__(self):
        """Gated Linear Unit with GELU activation, https://arxiv.org/abs/2002.05202."""
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * self.gelu(gates)


class SwiGLU(nn.Module):
    def __init__(self):
        """Gated Linear Unit with SiLU activation."""
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * self.silu(gates)


class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x) + 1.0


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class MinusAbs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -torch.abs(x)


def softabs(x):
    """https://stackoverflow.com/questions/49982438/how-to-restrict-the-output-of-neural-network-to-be-positive-in-python-keras"""
    return torch.where(torch.abs(x) < 1, 0.5 * torch.pow(x, 2.0), torch.abs(x) - 0.5)


class Softabs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softabs(x)


class MinusSoftabs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -softabs(x)


def get_activation(activation: str | None, **kwargs) -> nn.Module:
    """Helper function to get activation functions."""
    if activation is None or activation.lower() == "none":
        return nn.Identity()
    elif activation == "GeGLU":
        return GeGLU()
    elif activation == "ELUPlus":
        return ELUPlus()
    elif activation == "Abs":
        return Abs()
    elif activation == "MinusAbs":
        return MinusAbs()
    elif activation == "Softabs":
        return Softabs()
    elif activation == "MinusSoftabs":
        return MinusSoftabs()
    else:
        return getattr(nn, activation)(**kwargs)
