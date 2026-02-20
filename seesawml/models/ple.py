import hashlib
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from f9columnar.utils.helpers import load_pickle

from seesawml.models.activations import get_activation

# Minimum bin width to prevent division by zero
MIN_BIN_WIDTH = 1e-8

# Default tolerance for pad_value comparison (handles float precision issues)
DEFAULT_PAD_ATOL = 1e-6


def _compute_bin_fractions(
    x: torch.Tensor,
    bin_lower: torch.Tensor,
    bin_upper: torch.Tensor,
) -> torch.Tensor:
    """Compute fractional bin occupancy for input values.

    Parameters
    ----------
    x : torch.Tensor
        Input values of shape ``(batch_size,)`` or ``(batch_size, 1)``.
    bin_lower : torch.Tensor
        Lower bin edges of shape ``(n_bins,)``.
    bin_upper : torch.Tensor
        Upper bin edges of shape ``(n_bins,)``.

    Returns
    -------
    torch.Tensor
        Fractional bin occupancy of shape ``(batch_size, n_bins)``, clamped to [0, 1].
    """
    x_expanded = x.unsqueeze(-1) if x.dim() == 1 else x
    widths = bin_upper - bin_lower
    safe_widths = torch.clamp(widths, min=MIN_BIN_WIDTH)
    return torch.clamp((x_expanded - bin_lower) / safe_widths, 0.0, 1.0)


def _apply_padding_mask(
    z: torch.Tensor,
    x: torch.Tensor,
    pad_value: float | None,
    pad_atol: float,
) -> torch.Tensor:
    """Apply padding mask to bin encoding.

    Parameters
    ----------
    z : torch.Tensor
        Bin encoding of shape ``(batch_size, n_bins)``.
    x : torch.Tensor
        Original input values of shape ``(batch_size,)``.
    pad_value : float | None
        Padding value to detect. If None, no masking is applied.
    pad_atol : float
        Absolute tolerance for padding value comparison.

    Returns
    -------
    torch.Tensor
        Bin encoding with padding positions set to ``[0, 0, ..., 0, 1]``.
    """
    if pad_value is None:
        return z
    pad_mask = torch.abs(x - pad_value) < pad_atol
    pad_encoding = torch.zeros_like(z)
    pad_encoding[:, -1] = 1.0
    return torch.where(pad_mask.unsqueeze(-1), pad_encoding, z)


def _build_projection(n_bins: int, embedding_dim: int | None, act: str | None, bias: bool) -> nn.Module | None:
    """Build optional projection layer for PLE output.

    Parameters
    ----------
    n_bins : int
        Input dimension (number of bins).
    embedding_dim : int | None
        Output dimension. If None, no projection is created.
    act : str | None
        Activation function name. If None, no activation is applied.
    bias : bool
        Whether to include bias in linear layer.

    Returns
    -------
    nn.Module | None
        Sequential projection layer or None.
    """
    if embedding_dim is None:
        return None
    layers: list[nn.Module] = [nn.Linear(n_bins, embedding_dim, bias=bias)]
    if act is not None:
        layers.append(get_activation(act))
    return nn.Sequential(*layers)


class QuantilePiecewiseEncodingLayer(nn.Module):
    def __init__(
        self,
        ple_file_hash_str: str | None = None,
        feature_idx: int = 0,
        embedding_dim: int | None = None,
        act: str | None = None,
        bias: bool = True,
        dataset_key: str | None = None,
        quantile_bins: Sequence[float] | np.ndarray | None = None,
        pad_atol: float = DEFAULT_PAD_ATOL,
    ) -> None:
        """Quantile-based piecewise linear encoding layer.

        Parameters
        ----------
        ple_file_hash_str : str | None, optional
            Hash string to identify the quantile bins file. Used for training mode.
        feature_idx : int, optional
            Index of the feature to apply the piecewise encoding to, by default 0.
        embedding_dim : int | None, optional
            If specified, the output will be projected to this dimension, by default None.
        act : str | None, optional
            Activation function to apply after the projection layer, by default None.
        bias : bool, optional
            If True, the projection layer will have a bias term, by default True.
        dataset_key : str | None, optional
            If specified, the dataset key to use for loading quantile bins, by default None.
        quantile_bins : Sequence[float] | np.ndarray | None, optional
            Pre-computed quantile bins to embed directly. Used for ONNX export mode.
            If provided, ple_file_hash_str is ignored.
        pad_atol : float, optional
            Absolute tolerance for pad_value comparison. Default is 1e-6.

        Notes
        -----
        Either `ple_file_hash_str` or `quantile_bins` must be provided.
        Run `calculate_quantiles` to generate the quantile bins file before using this layer.
        Bins should be finite values only (no -inf/inf). Last output bin is reserved for padding
        (outputs all zeros except 1 in the last position for padded inputs).
        """
        super().__init__()
        self.pad_atol = pad_atol

        if quantile_bins is not None:
            # ONNX mode: use provided bins directly
            bins_tensor = torch.tensor(quantile_bins, dtype=torch.float32)
        elif ple_file_hash_str is not None:
            # Training mode: load from pickle file
            quantile_bins_lst, _ = self._load_quantile_bins(ple_file_hash_str, dataset_key)
            bins_tensor = torch.tensor(quantile_bins_lst[feature_idx], dtype=torch.float32)
        else:
            raise ValueError("Either ple_file_hash_str or quantile_bins must be provided")

        # Strip inf edges if present (backwards compatibility with old bin files)
        if len(bins_tensor) > 0 and bins_tensor[0] == float("-inf"):
            bins_tensor = bins_tensor[1:]
        if len(bins_tensor) > 0 and bins_tensor[-1] == float("inf"):
            bins_tensor = bins_tensor[:-1]

        # Use register_buffer for ONNX compatibility (no gradients needed)
        # Stored format: [b0, b1, ..., b(n-1)] (finite boundaries only)
        self.register_buffer("_finite_bins", bins_tensor)
        # n_bins = number of output bins = len(finite_bins) (intervals between boundaries) + 1 (padding bin)
        # With n boundaries, we have n-1 intervals, plus 1 padding bin = n output bins
        self.n_bins = len(bins_tensor)

        self.projection = _build_projection(self.n_bins, embedding_dim, act, bias)

    def _load_quantile_bins(
        self, ple_file_hash_str: str, dataset_key: str | None = None
    ) -> tuple[list[np.ndarray], None]:
        """Load quantile bin boundaries from pickle file.

        Parameters
        ----------
        ple_file_hash_str : str
            String to hash for locating the pickle file.
        dataset_key : str | None
            Key to select dataset within the pickle. Defaults to "events".

        Returns
        -------
        tuple[list[np.ndarray], None]
            List of bin edge arrays per feature. Second element is None (deprecated).

        Notes
        -----
        Pickle file is located at $ANALYSIS_ML_RESULTS_DIR/quantile_bins/{md5_hash}.p
        """
        quantile_bins_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "quantile_bins")
        quantile_bins_file_name = hashlib.md5(ple_file_hash_str.encode()).hexdigest()

        quantile_bins_file_path = os.path.join(quantile_bins_dir, f"{quantile_bins_file_name}.p")

        quantile_bins_dct = load_pickle(quantile_bins_file_path)

        if dataset_key is not None:
            ds_quantile_bins_lst = quantile_bins_dct[dataset_key]
        else:
            ds_quantile_bins_lst = quantile_bins_dct["events"]

        return ds_quantile_bins_lst, None

    def forward(self, x: torch.Tensor, pad_value: float | None = None) -> torch.Tensor:
        """Compute piecewise linear encoding of input values.

        Maps continuous values to bin occupancy vectors using quantile-based
        boundaries. Values below first bin get 0 in all bins, values above last
        bin get 1 in all bins. Last bin is reserved for padding indicator.

        Parameters
        ----------
        x : torch.Tensor
            Input values of shape ``(batch_size,)`` or ``(batch_size, seq_len)``.
        pad_value : float or None, optional
            If provided, positions where ``x == pad_value`` are encoded as
            padding (all zeros except last bin = 1).

        Returns
        -------
        torch.Tensor
            Bin encoding of shape ``(batch_size, n_bins)`` with values in [0, 1].
            NaN inputs are treated as 0 to prevent propagation.
        """
        batch_size = x.size(0)

        # Handle NaN early: treat as zero (will be overwritten if padding)
        x_safe = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # _finite_bins: [b0, b1, ..., b(n-1)] (finite boundaries only)
        # Output bins: fractional position between consecutive boundaries
        # Last output bin: reserved for padding (always 0 for real data, 1 for padding)

        n_boundaries = len(self._finite_bins)
        if n_boundaries > 1:
            # Compute fractional bin occupancy for each interval
            bin_lower = self._finite_bins[:-1]
            bin_upper = self._finite_bins[1:]
            frac = _compute_bin_fractions(x_safe, bin_lower, bin_upper)
        else:
            frac = x.new_empty(batch_size, 0)

        # Add last bin (padding indicator, always 0 for real data)
        last_bin = x.new_zeros(batch_size, 1)
        z = torch.cat([frac, last_bin], dim=1)

        z = _apply_padding_mask(z, x, pad_value, self.pad_atol)

        if self.projection is not None:
            z = self.projection(z)

        return z


class LearnablePiecewiseEncodingLayer(nn.Module):
    def __init__(
        self,
        n_bins: int,
        embedding_dim: int | None = None,
        act: str | None = None,
        learn_bins: bool = True,
        bias: bool = True,
        pad_atol: float = DEFAULT_PAD_ATOL,
    ) -> None:
        """Learnable piecewise linear encoding layer.

        Source: https://github.com/OpenTabular/DeepTabular/blob/master/mambular/arch_utils/learnable_ple.py
        Reference: https://arxiv.org/abs/2203.05556

        Parameters
        ----------
        n_bins : int
            Number of bins for the piecewise encoding.
        embedding_dim : int | None, optional
            If specified, the output will be projected to this dimension, by default None.
        act : str | None, optional
            Activation function to apply after the projection layer, by default None.
        learn_bins : bool, optional
            If True, the bin boundaries are learnable parameters, by default True.
        bias : bool, optional
            If True, the projection layer will have a bias term, by default True.
        pad_atol : float, optional
            Absolute tolerance for pad_value comparison. Default is 1e-6.

        Note
        ----
        Features should be normalized to [0, 1] before applying this layer. The first bin edge is 0, the last bin edge
        is 1, no out of range values are allowed.

        """
        super().__init__()
        self.n_bins = n_bins
        self.learn_bins = learn_bins
        self.pad_atol = pad_atol

        if self.learn_bins:
            init_bins = torch.linspace(0, 1, n_bins + 1)
            self.bin_boundaries_param = nn.Parameter(init_bins)
        else:
            fixed_bins = torch.linspace(0, 1, n_bins + 1)
            self.register_buffer("bin_boundaries", fixed_bins)

        self.projection = _build_projection(self.n_bins, embedding_dim, act, bias)

    def forward(self, x: torch.Tensor, pad_value: float | None = None) -> torch.Tensor:
        """Compute piecewise linear encoding of input values.

        Parameters
        ----------
        x : torch.Tensor
            Input values of shape (batch_size,), normalized to [0, 1].
        pad_value : float | None
            If provided, positions where x == pad_value are encoded as
            padding (all zeros except last bin = 1).

        Returns
        -------
        torch.Tensor
            Bin encoding of shape (batch_size, n_bins) with values in [0, 1].
        """
        if self.learn_bins:
            sorted_bins = torch.sort(self.bin_boundaries_param)[0]
        else:
            sorted_bins = self.bin_boundaries

        # Handle NaN early
        x_safe = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        z = _compute_bin_fractions(x_safe, sorted_bins[:-1], sorted_bins[1:])
        z = _apply_padding_mask(z, x, pad_value, self.pad_atol)

        if self.projection is not None:
            z = self.projection(z)

        return z
