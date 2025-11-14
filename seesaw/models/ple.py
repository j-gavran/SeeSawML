import hashlib
import os

import torch
import torch.nn as nn
from f9columnar.utils.helpers import load_pickle

from seesaw.models.activations import get_activation


class QuantilePiecewiseEncodingLayer(nn.Module):
    def __init__(
        self,
        ple_file_hash_str: str,
        feature_idx: int,
        embedding_dim: int | None = None,
        act: str | None = None,
        bias: bool = True,
        dataset_key: str | None = None,
    ) -> None:
        """Quantile-based piecewise linear encoding layer.

        Parameters
        ----------
        ple_file_hash_str : str
            Hash string to identify the quantile bins file.
        feature_idx : int
            Index of the feature to apply the piecewise encoding to.
        embedding_dim : int | None, optional
            If specified, the output will be projected to this dimension, by default None.
        act : str | None, optional
            Activation function to apply after the projection layer, by default None.
        bias : bool, optional
            If True, the projection layer will have a bias term, by default True.
        dataset_key : str | None, optional
            If specified, the dataset key to use for loading quantile bins, by default None.

        Note
        ----
        Run `calculate_quantiles` to generate the quantile bins file before using this layer. First bin edge is -inf,
        last bin edge is inf, and the last bin is reserved for padding values.

        """
        super().__init__()
        quantile_bins_lst, n_bins_lst = self._load_quantile_bins(ple_file_hash_str, dataset_key)

        self.qunatile_bins_param = nn.Parameter(
            torch.tensor(quantile_bins_lst[feature_idx], dtype=torch.float32), requires_grad=False
        )
        self.bins = n_bins_lst[feature_idx]

        self.projection: nn.Module | None

        if embedding_dim is not None:
            projection: list[nn.Module] = [nn.Linear(self.bins, embedding_dim, bias=bias)]
            projection.append(get_activation(act))
            self.projection = nn.Sequential(*projection)
        else:
            self.projection = None

    def _load_quantile_bins(
        self, ple_file_hash_str: str, dataset_key: str | None = None
    ) -> tuple[list[float], list[int]]:
        quantile_bins_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "quantile_bins")
        quantile_bins_file_name = hashlib.md5(ple_file_hash_str.encode()).hexdigest()

        quantile_bins_file_path = os.path.join(quantile_bins_dir, f"{quantile_bins_file_name}.p")

        quantile_bins_dct = load_pickle(quantile_bins_file_path)

        if dataset_key is not None:
            ds_quantile_bins_lst = quantile_bins_dct[dataset_key]
        else:
            ds_quantile_bins_lst = quantile_bins_dct["events"]

        return ds_quantile_bins_lst, [len(bins) - 1 for bins in ds_quantile_bins_lst]

    def forward(self, x: torch.Tensor, pad_value: float | None = None) -> torch.Tensor:
        # prepare output tensor
        z = torch.zeros(x.size(0), self.bins, dtype=x.dtype, device=x.device)

        # compute padding mask once
        pad_mask = (x == pad_value) if pad_value is not None else None

        # expand x and bin boundaries for vectorized comparison
        # b_t_1, b_t: [1, bins]
        b_t_1 = self.qunatile_bins_param[:-1]
        b_t = self.qunatile_bins_param[1:]

        # x_expanded: [batch, bins]
        x_expanded = x.unsqueeze(-1)

        # compute interpolation weights vectorized
        # (x >= b_t_1) & (x < b_t) selects the correct interval
        frac = (x_expanded - b_t_1) / (b_t - b_t_1)

        frac[:, 0] = 1  # edge case: x < first bin, always valid
        frac[:, -1] = 0  # edge case: x >= last bin, reserved for padding

        frac = torch.clamp(frac, 0, 1)  # ensure valid range

        # assign 0/1 outside the bins (vectorized)
        z = torch.where(x_expanded < b_t_1, 0, frac)
        z = torch.where(x_expanded >= b_t, 1, z)

        # handle padding values
        if pad_mask is not None:
            z[pad_mask, :] = 0
            z[pad_mask, -1] = 1  # last bin = pad indicator

        # optional projection
        if self.projection is not None:
            z = self.projection(z)

        return z


class LearnablePiecewiseEncodingLayer(nn.Module):
    def __init__(
        self,
        bins: int,
        embedding_dim: int | None = None,
        act: str | None = None,
        learn_bins: bool = True,
        bias: bool = True,
    ) -> None:
        """Learnable piecewise linear encoding layer.

        Source: https://github.com/OpenTabular/DeepTabular/blob/master/mambular/arch_utils/learnable_ple.py
        Reference: https://arxiv.org/abs/2203.05556

        Parameters
        ----------
        bins : int
            Number of bins for the periodic encoding.
        embedding_dim : int | None, optional
            If specified, the output will be projected to this dimension, by default None.
        act : str | None, optional
            Activation function to apply after the projection layer, by default None.
        learn_bins : bool, optional
            If True, the bin boundaries are learnable parameters, by default True.
        bias : bool, optional
            If True, the projection layer will have a bias term, by default True.

        Note
        ----
        Features should be normalized to [0, 1] before applying this layer. The first bin edge is 0, the last bin edge
        is 1, no out of range values are allowed.

        """
        super().__init__()
        self.bins = bins
        self.learn_bins = learn_bins

        if self.learn_bins:
            init_bins = torch.linspace(0, 1, bins + 1)
            self.bin_boundaries_param = nn.Parameter(init_bins)
        else:
            self.bin_boundaries: torch.Tensor

            fixed_bins = torch.linspace(0, 1, bins + 1)
            self.register_buffer("bin_boundaries", fixed_bins)

        self.projection: nn.Module | None
        if embedding_dim is not None:
            projection: list[nn.Module] = [nn.Linear(self.bins, embedding_dim, bias=bias)]
            projection.append(get_activation(act))
            self.projection = nn.Sequential(*projection)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor, pad_value: float | None = None) -> torch.Tensor:
        if self.learn_bins:
            sorted_bins = torch.sort(self.bin_boundaries_param)[0]
        else:
            sorted_bins = self.bin_boundaries

        z = torch.zeros(x.size(0), self.bins, dtype=x.dtype, device=x.device)

        pad_mask = (x == pad_value) if pad_value is not None else None

        b_t_1 = sorted_bins[:-1]
        b_t = sorted_bins[1:]

        x_expanded = x.unsqueeze(-1)

        frac = (x_expanded - b_t_1) / (b_t - b_t_1)
        frac = torch.clamp(frac, 0, 1)

        z = torch.where(x_expanded < b_t_1, 0, frac)
        z = torch.where(x_expanded >= b_t, 1, z)

        if pad_mask is not None:
            z[pad_mask, :] = 0
            z[pad_mask, -1] = 1

        if self.projection is not None:
            z = self.projection(z)

        return z
