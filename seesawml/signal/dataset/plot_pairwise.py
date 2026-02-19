"""Standalone CLI script for pairwise feature plotting.

Computes and plots ParT-style pairwise features (delta_r, kt, z, m2) directly from HDF5 data.
Optionally computes attention gradient importance if a model checkpoint is provided.

Usage:
    plot_pairwise
    plot_pairwise 50000
    plot_pairwise -c signal/2L -n 100000
    plot_pairwise -c signal/2L -m /path/to/model.ckpt
"""

import argparse
import copy
import logging
import os
import traceback
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from einops import rearrange
from f9columnar.ml.dataloader_helpers import get_hdf5_metadata
from f9columnar.ml.hdf5_dataloader import full_collate_fn, get_ml_hdf5_dataloader
from matplotlib.lines import Line2D
from omegaconf import DictConfig, open_dict
from omegaconf import DictConfig as OmegaDictConfig
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from seesawml.models.nn_modules import BaseLightningModule
from seesawml.models.transformers.attention import build_adjacency_attention_mask
from seesawml.models.transformers.pairwise_features import (
    DEFAULT_PARTICLE_MASSES,
    OBJECTS_SHORT_NAMES,
    PairwiseFeaturesCalculator,
    derive_energy_from_mass,
)
from seesawml.models.transformers.particle_transformer import ParticleTransformer
from seesawml.signal.training.sig_bkg_trainer import load_sig_bkg_model
from seesawml.signal.utils import get_classifier_labels
from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.hydra_initalize import get_hydra_config
from seesawml.utils.loggers import setup_logger
from seesawml.utils.plots_utils import save_plot

PAIRWISE_QUANTITIES = ("delta_r", "kt", "z", "m2")

QUANTITY_LATEX = {
    "delta_r": r"$\ln \Delta R$",
    "kt": r"$\ln k_T$",
    "z": r"$\ln z$",
    "m2": r"$\ln m^2$",
}

QUANTITY_UNITS = {
    "delta_r": "",
    "kt": "[GeV]",
    "z": "",
    "m2": r"[GeV$^2$]",
}

OBJECT_LABEL_PREFIX = {"jets": "j", "electrons": "e", "muons": "μ", "taus": "τ"}

DATASET_SHORTHAND = {
    "0L": "ttHcc_0L_flat",
    "1L": "ttHcc_1L_flat",
    "2L": "ttHcc_2L_jagged",
}


@dataclass
class ObjectSpec:
    """Specification for a single object type (jets, electrons, etc.)."""

    name: str
    n_objects: int
    pad_value: float | None
    rest_mass: float | None
    pt_idx: int
    eta_idx: int
    phi_idx: int
    e_idx: int | None


@dataclass
class PairwiseAccumulator:
    """Accumulator for streaming pairwise feature statistics."""

    feature_sums: dict[str, np.ndarray] = field(default_factory=dict)
    pair_counts: dict[str, np.ndarray] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    feature_samples: dict[str, list[np.ndarray]] = field(default_factory=dict)
    max_samples: int = 500_000

    def _init_class(self, class_name: str, feat_sum: np.ndarray, cnt_sum: np.ndarray) -> None:
        """Initialize storage for a new class."""
        self.feature_sums[class_name] = np.zeros_like(feat_sum, dtype=np.float64)
        self.pair_counts[class_name] = np.zeros_like(cnt_sum, dtype=np.int64)
        self.event_counts[class_name] = 0
        self.feature_samples[class_name] = []

    def add_batch(
        self,
        class_name: str,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        n_events: int,
    ) -> None:
        """Add a batch of pairwise features for a class."""
        valid_f = valid_mask.unsqueeze(-1).to(dtype=features.dtype)
        feat_sum = (features * valid_f).sum(dim=0).cpu().numpy().astype(np.float64)
        cnt_sum = valid_mask.sum(dim=0).cpu().numpy().astype(np.int64)

        if class_name not in self.feature_sums:
            self._init_class(class_name, feat_sum, cnt_sum)

        self.feature_sums[class_name] += feat_sum
        self.pair_counts[class_name] += cnt_sum
        self.event_counts[class_name] += n_events

        self._store_samples(class_name, features, valid_mask)

    def _store_samples(self, class_name: str, features: torch.Tensor, valid_mask: torch.Tensor) -> None:
        """Store raw samples for 1D/2D plots."""
        current = sum(s.shape[0] for s in self.feature_samples[class_name])
        if current >= self.max_samples:
            return

        valid_flat = valid_mask.reshape(-1)
        feat_flat = features.reshape(-1, features.shape[-1])
        valid_feats = feat_flat[valid_flat].cpu().numpy()

        remaining = self.max_samples - current
        if valid_feats.shape[0] > remaining:
            idx = np.random.choice(valid_feats.shape[0], remaining, replace=False)
            valid_feats = valid_feats[idx]

        if valid_feats.shape[0] > 0:
            self.feature_samples[class_name].append(valid_feats)

    def get_count(self, class_name: str) -> int:
        """Get event count for a class."""
        return self.event_counts.get(class_name, 0)

    def is_done(self, class_labels: dict[str, int], target: int) -> bool:
        """Check if we've collected enough events for all classes."""
        return all(self.get_count(name) >= target for name in class_labels)

    def finalize(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, int],
    ]:
        """Compute final averages and return results."""
        avg_features: dict[str, np.ndarray] = {}
        presence: dict[str, np.ndarray] = {}
        samples: dict[str, np.ndarray] = {}

        for class_name, feat_sum in self.feature_sums.items():
            n_events = self.event_counts.get(class_name, 0)
            if n_events <= 0:
                continue

            pair_cnt = self.pair_counts[class_name]
            denom = pair_cnt[..., None].astype(np.float64)

            with np.errstate(invalid="ignore", divide="ignore"):
                avg = np.divide(
                    feat_sum,
                    denom,
                    out=np.full_like(feat_sum, np.nan, dtype=np.float64),
                    where=denom != 0.0,
                )
            avg_features[class_name] = avg.astype(np.float32)
            presence[class_name] = (pair_cnt / float(n_events) * 100.0).astype(np.float32)

            if self.feature_samples[class_name]:
                samples[class_name] = np.concatenate(self.feature_samples[class_name])

        return avg_features, presence, samples, dict(self.event_counts)


@dataclass
class GradientAccumulator:
    """Accumulator for attention gradient statistics."""

    gradient_sums: dict[str, np.ndarray] = field(default_factory=dict)
    gradient_sq_sums: dict[str, np.ndarray] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)

    def add_batch(self, class_name: str, gradients: np.ndarray, n_events: int) -> None:
        """Add gradient batch for a class."""
        if class_name not in self.gradient_sums:
            self.gradient_sums[class_name] = np.zeros_like(gradients, dtype=np.float64)
            self.gradient_sq_sums[class_name] = np.zeros_like(gradients, dtype=np.float64)
            self.event_counts[class_name] = 0

        grads_f64 = gradients.astype(np.float64)
        self.gradient_sums[class_name] += grads_f64 * n_events
        self.gradient_sq_sums[class_name] += (grads_f64**2) * n_events
        self.event_counts[class_name] += n_events

    def get_count(self, class_name: str) -> int:
        return self.event_counts.get(class_name, 0)

    def is_done(self, class_labels: dict[str, int], target: int) -> bool:
        return all(self.get_count(name) >= target for name in class_labels)

    def finalize(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
        """Compute mean and std of gradients per class."""
        mean_grads: dict[str, np.ndarray] = {}
        std_grads: dict[str, np.ndarray] = {}

        for class_name, grad_sum in self.gradient_sums.items():
            n = self.event_counts[class_name]
            if n <= 0:
                continue

            mean = grad_sum / n
            mean_grads[class_name] = mean.astype(np.float32)

            var = self.gradient_sq_sums[class_name] / n - mean**2
            std_grads[class_name] = np.sqrt(np.maximum(var, 0)).astype(np.float32)

        return mean_grads, std_grads, dict(self.event_counts)


def _get_grid_layout(n_plots: int) -> tuple[int, int, tuple[int, int]]:
    """Get grid layout (nrows, ncols, figsize) for n_plots."""
    layouts = {
        1: (1, 1, (8, 7)),
        2: (1, 2, (14, 6)),
        3: (1, 3, (18, 6)),
        4: (2, 2, (14, 12)),
    }
    return layouts.get(n_plots, (2, 2, (14, 12)))


def _get_valid_classes(counts: dict[str, int], data: dict[str, np.ndarray]) -> list[str]:
    """Get list of classes with data."""
    return [c for c, n in counts.items() if n > 0 and c in data]


def _get_colors(n: int) -> list:
    """Get color palette for n classes."""
    return list(plt.cm.tab10.colors[:n])  # type: ignore[attr-defined]


def _add_object_boundaries(ax: plt.Axes, boundaries: list[int], color: str = "white") -> None:
    """Add boundary lines between object types."""
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color=color, linewidth=1.5, alpha=0.7)
        ax.axvline(b - 0.5, color=color, linewidth=1.5, alpha=0.7)


def _mask_diagonal(data: np.ndarray) -> np.ndarray:
    """Return a copy with diagonal entries set to NaN."""
    if data.ndim < 2:
        return data
    masked = data.copy()
    n = min(masked.shape[0], masked.shape[1])
    idx = np.arange(n)
    masked[idx, idx, ...] = np.nan
    return masked


def _compute_color_range(
    data: np.ndarray, symmetric: bool = False, percentiles: tuple[float, float] = (5, 95)
) -> tuple[float, float]:
    """Compute color range from data."""
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)

    if symmetric:
        vmax = float(np.nanpercentile(np.abs(valid), percentiles[1]))
        return -vmax, vmax

    vmin, vmax = np.nanpercentile(valid, list(percentiles)).astype(float)
    return float(vmin), float(vmax)


def plot_heatmap_grid(
    data: np.ndarray,
    object_labels: list[str],
    object_boundaries: list[int],
    title: str,
    save_path: str,
    filename: str,
    quantities: tuple[str, ...] = PAIRWISE_QUANTITIES,
    cmap: str = "viridis",
    symmetric: bool = False,
    feature_ranges: list[tuple[float, float]] | None = None,
) -> None:
    """Plot a grid of heatmaps for pairwise features."""
    n_features = len(quantities)
    nrows, ncols, figsize = _get_grid_layout(n_features)
    data = _mask_diagonal(data)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgrey")

    for i, (ax, qty) in enumerate(zip(axes[:n_features], quantities)):
        feat = data[:, :, i]

        if feature_ranges and i < len(feature_ranges):
            vmin, vmax = feature_ranges[i]
        else:
            vmin, vmax = _compute_color_range(feat, symmetric)

        im = ax.imshow(np.ma.masked_invalid(feat), cmap=cmap_obj, aspect="equal", vmin=vmin, vmax=vmax)
        _add_object_boundaries(ax, object_boundaries, "black" if symmetric else "white")

        ax.set_xticks(range(len(object_labels)))
        ax.set_yticks(range(len(object_labels)))
        ax.set_xticklabels(object_labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(object_labels, fontsize=8)
        ax.set_title(QUANTITY_LATEX.get(qty, qty), fontsize=13)
        ax.set_xlabel("Object j", fontsize=11)
        ax.set_ylabel("Object i", fontsize=11)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=9)

    for ax in axes[n_features:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, y=0.995)
    plt.tight_layout()
    save_plot(fig, f"{save_path}/{filename}.pdf", use_format="pdf")


def plot_presence_heatmap(
    data: np.ndarray,
    object_labels: list[str],
    object_boundaries: list[int],
    title: str,
    save_path: str,
    filename: str,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> None:
    """Plot a single presence heatmap."""
    data = _mask_diagonal(data)
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgrey")

    if symmetric:
        vmin, vmax = _compute_color_range(data, symmetric=True)
    else:
        vmin, vmax = 0.0, 100.0

    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap_obj, aspect="equal", vmin=vmin, vmax=vmax)
    _add_object_boundaries(ax, object_boundaries, "black" if symmetric else "white")

    ax.set_xticks(range(len(object_labels)))
    ax.set_yticks(range(len(object_labels)))
    ax.set_xticklabels(object_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(object_labels, fontsize=8)
    ax.set_xlabel("Object j", fontsize=11)
    ax.set_ylabel("Object i", fontsize=11)
    ax.set_title(title, fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pair presence [%]" if not symmetric else "Δ presence [pp]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    save_plot(fig, f"{save_path}/{filename}.pdf", use_format="pdf")


def plot_1d_distributions(
    samples: dict[str, np.ndarray],
    counts: dict[str, int],
    save_path: str,
    quantities: tuple[str, ...] = PAIRWISE_QUANTITIES,
    n_bins: int = 80,
) -> None:
    """Plot 1D distribution histograms for each pairwise feature."""
    class_names = _get_valid_classes(counts, samples)
    if not class_names:
        return

    colors = _get_colors(len(class_names))
    n_features = len(quantities)
    nrows, ncols, figsize = _get_grid_layout(n_features)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, (ax, qty) in enumerate(zip(axes[:n_features], quantities)):
        all_vals = [
            samples[c][:, i][np.isfinite(samples[c][:, i])]
            for c in class_names
            if samples[c][:, i][np.isfinite(samples[c][:, i])].size > 0
        ]

        if not all_vals:
            ax.set_visible(False)
            continue

        all_concat = np.concatenate(all_vals)
        bins = np.linspace(np.min(all_concat), np.max(all_concat), n_bins + 1)

        for c, color in zip(class_names, colors):
            vals = samples[c][:, i]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            ax.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"{c} (n={counts[c]:,})",
                color=color,
                histtype="stepfilled",
                linewidth=1.2,
                edgecolor=color,
            )

        ax.set_xlabel(f"{QUANTITY_LATEX.get(qty, qty)} {QUANTITY_UNITS.get(qty, '')}", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(QUANTITY_LATEX.get(qty, qty), fontsize=13)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=9)

    for ax in axes[n_features:]:
        ax.set_visible(False)

    fig.suptitle("Pairwise Feature Distributions", fontsize=15, y=0.995)
    plt.tight_layout()
    save_plot(fig, f"{save_path}/pairwise_1d_distributions.pdf", use_format="pdf")


def plot_2d_correlations(
    samples: dict[str, np.ndarray],
    counts: dict[str, int],
    save_path: str,
    quantities: tuple[str, ...] = PAIRWISE_QUANTITIES,
    max_points: int = 50_000,
) -> None:
    """Plot 2D scatter plots for pairwise feature correlations."""
    class_names = _get_valid_classes(counts, samples)
    if not class_names:
        return

    qty_idx = {q: i for i, q in enumerate(quantities)}
    pairs = [("delta_r", "kt"), ("delta_r", "z"), ("kt", "z"), ("delta_r", "m2")]
    pairs = [(p1, p2) for p1, p2 in pairs if p1 in qty_idx and p2 in qty_idx]

    if not pairs:
        return

    colors = _get_colors(len(class_names))

    for q1, q2 in pairs:
        idx1, idx2 = qty_idx[q1], qty_idx[q2]
        fig, ax = plt.subplots(figsize=(9, 8))

        for c, color in zip(class_names, colors):
            v1, v2 = samples[c][:, idx1], samples[c][:, idx2]
            mask = np.isfinite(v1) & np.isfinite(v2)
            v1, v2 = v1[mask], v2[mask]

            if v1.size == 0:
                continue

            if v1.size > max_points:
                idx = np.random.choice(v1.size, max_points, replace=False)
                v1, v2 = v1[idx], v2[idx]

            ax.scatter(v1, v2, s=3, alpha=0.3, label=c, color=color, rasterized=True)

        ax.set_xlabel(f"{QUANTITY_LATEX.get(q1, q1)} {QUANTITY_UNITS.get(q1, '')}", fontsize=12)
        ax.set_ylabel(f"{QUANTITY_LATEX.get(q2, q2)} {QUANTITY_UNITS.get(q2, '')}", fontsize=12)
        ax.set_title(f"{QUANTITY_LATEX.get(q1, q1)} vs {QUANTITY_LATEX.get(q2, q2)}", fontsize=14)
        ax.legend(fontsize=10, markerscale=3)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        save_plot(fig, f"{save_path}/pairwise_2d_{q1}_vs_{q2}.pdf", use_format="pdf")


def plot_2d_density(
    samples: dict[str, np.ndarray],
    counts: dict[str, int],
    save_path: str,
    quantities: tuple[str, ...] = PAIRWISE_QUANTITIES,
    n_bins: int = 60,
) -> None:
    """Plot 2D density contour plots comparing classes."""
    class_names = _get_valid_classes(counts, samples)
    if len(class_names) < 2:
        return

    qty_idx = {q: i for i, q in enumerate(quantities)}
    pairs = [("delta_r", "kt"), ("delta_r", "z")]
    pairs = [(p1, p2) for p1, p2 in pairs if p1 in qty_idx and p2 in qty_idx]

    if not pairs:
        return

    colors = _get_colors(len(class_names))

    for q1, q2 in pairs:
        idx1, idx2 = qty_idx[q1], qty_idx[q2]
        fig, ax = plt.subplots(figsize=(9, 8))

        # Compute global range
        all_v1, all_v2 = [], []
        for c in class_names:
            v1, v2 = samples[c][:, idx1], samples[c][:, idx2]
            mask = np.isfinite(v1) & np.isfinite(v2)
            all_v1.append(v1[mask])
            all_v2.append(v2[mask])

        v1_cat, v2_cat = np.concatenate(all_v1), np.concatenate(all_v2)
        x_range = np.percentile(v1_cat, [2, 98])
        y_range = np.percentile(v2_cat, [2, 98])

        for c, color in zip(class_names, colors):
            v1, v2 = samples[c][:, idx1], samples[c][:, idx2]
            mask = np.isfinite(v1) & np.isfinite(v2)
            v1, v2 = v1[mask], v2[mask]

            if v1.size < 100:
                continue

            h, xedges, yedges = np.histogram2d(v1, v2, bins=n_bins, range=[x_range, y_range], density=True)
            h = gaussian_filter(h.T, sigma=1.0)

            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])

            levels = np.percentile(h[h > 0], [50, 75, 90, 99]) if h.max() > 0 else [0.1, 0.5, 1.0]
            ax.contour(xc, yc, h, levels=levels, colors=[color], linewidths=1.5, alpha=0.8)
            ax.contourf(xc, yc, h, levels=levels, colors=[color], alpha=0.15)

        handles = [Line2D([0], [0], color=c, lw=2, label=n) for n, c in zip(class_names, colors)]
        ax.legend(handles=handles, fontsize=10)

        ax.set_xlabel(f"{QUANTITY_LATEX.get(q1, q1)} {QUANTITY_UNITS.get(q1, '')}", fontsize=12)
        ax.set_ylabel(f"{QUANTITY_LATEX.get(q2, q2)} {QUANTITY_UNITS.get(q2, '')}", fontsize=12)
        ax.set_title(f"Density: {QUANTITY_LATEX.get(q1, q1)} vs {QUANTITY_LATEX.get(q2, q2)}", fontsize=14)
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        save_plot(fig, f"{save_path}/pairwise_2d_density_{q1}_vs_{q2}.pdf", use_format="pdf")


def _compute_feature_ranges(pairwise_avg: dict[str, np.ndarray], class_names: list[str]) -> list[tuple[float, float]]:
    """Compute global feature ranges across all classes."""
    ranges = []
    for i in range(len(PAIRWISE_QUANTITIES)):
        vals = []
        for c in class_names:
            arr = _mask_diagonal(pairwise_avg[c])[:, :, i]
            v = arr[np.isfinite(arr)]
            if v.size:
                vals.append(v.astype(np.float64))

        if vals:
            all_v = np.concatenate(vals)
            vmin, vmax = np.nanpercentile(all_v, [5, 95]).astype(float)
            ranges.append((float(vmin), float(vmax)))
        else:
            ranges.append((-1.0, 1.0))
    return ranges


def _plot_class_heatmaps(
    pairwise_avg: dict[str, np.ndarray],
    presence: dict[str, np.ndarray],
    counts: dict[str, int],
    class_names: list[str],
    object_labels: list[str],
    object_boundaries: list[int],
    save_path: str,
    feature_ranges: list[tuple[float, float]],
) -> None:
    """Plot individual class heatmaps."""
    for c in class_names:
        plot_heatmap_grid(
            pairwise_avg[c],
            object_labels,
            object_boundaries,
            f"Pairwise Features: {c} (n={counts[c]:,})",
            save_path,
            f"pairwise_{c}",
            feature_ranges=feature_ranges,
        )
        plot_presence_heatmap(
            presence[c],
            object_labels,
            object_boundaries,
            f"Pair Presence: {c} (n={counts[c]:,})",
            save_path,
            f"pairwise_presence_{c}",
        )


def _plot_difference_heatmaps(
    pairwise_avg: dict[str, np.ndarray],
    presence: dict[str, np.ndarray],
    counts: dict[str, int],
    class_names: list[str],
    object_labels: list[str],
    object_boundaries: list[int],
    save_path: str,
) -> None:
    """Plot difference heatmaps (one-vs-rest and pairwise)."""
    if len(class_names) < 2:
        return

    total = sum(counts[c] for c in class_names)
    overall_avg = sum(np.nan_to_num(pairwise_avg[c], nan=0.0) * counts[c] for c in class_names) / total
    overall_pres = sum(presence[c] * counts[c] for c in class_names) / total

    # One-vs-rest
    for c in class_names:
        rest_count = total - counts[c]
        if rest_count == 0:
            continue

        rest_avg = (overall_avg * total - np.nan_to_num(pairwise_avg[c], nan=0.0) * counts[c]) / rest_count
        rest_pres = (overall_pres * total - presence[c] * counts[c]) / rest_count

        with np.errstate(invalid="ignore", divide="ignore"):
            diff = 100.0 * (pairwise_avg[c] - rest_avg) / rest_avg

        plot_heatmap_grid(
            diff,
            object_labels,
            object_boundaries,
            f"Pairwise Δ%: {c} vs rest",
            save_path,
            f"pairwise_diffpct_{c}_vs_rest",
            cmap="RdBu_r",
            symmetric=True,
        )
        plot_presence_heatmap(
            presence[c] - rest_pres,
            object_labels,
            object_boundaries,
            f"Pair Presence Δpp: {c} vs rest",
            save_path,
            f"pairwise_presence_diff_{c}_vs_rest",
            cmap="RdBu_r",
            symmetric=True,
        )

    # Pairwise comparisons
    for i, c1 in enumerate(class_names):
        for c2 in class_names[i + 1 :]:
            with np.errstate(invalid="ignore", divide="ignore"):
                diff = 100.0 * (pairwise_avg[c1] - pairwise_avg[c2]) / pairwise_avg[c2]

            plot_heatmap_grid(
                diff,
                object_labels,
                object_boundaries,
                f"Pairwise Δ%: {c1} vs {c2}",
                save_path,
                f"pairwise_diffpct_{c1}_vs_{c2}",
                cmap="RdBu_r",
                symmetric=True,
            )
            plot_presence_heatmap(
                presence[c1] - presence[c2],
                object_labels,
                object_boundaries,
                f"Pair Presence Δpp: {c1} vs {c2}",
                save_path,
                f"pairwise_presence_diff_{c1}_vs_{c2}",
                cmap="RdBu_r",
                symmetric=True,
            )


def _plot_group_heatmaps(
    pairwise_avg: dict[str, np.ndarray],
    presence: dict[str, np.ndarray],
    counts: dict[str, int],
    custom_groups: dict[str, list[str]],
    object_labels: list[str],
    object_boundaries: list[int],
    save_path: str,
    feature_ranges: list[tuple[float, float]],
) -> None:
    """Plot custom group heatmaps."""
    if not custom_groups or len(custom_groups) < 2:
        return

    group_avgs: dict[str, np.ndarray] = {}
    group_pres: dict[str, np.ndarray] = {}
    group_counts: dict[str, int] = {}

    for gname, members in custom_groups.items():
        valid = [m for m in members if m in pairwise_avg and counts.get(m, 0) > 0]
        if not valid:
            continue

        gc = sum(counts[m] for m in valid)
        if gc == 0:
            continue

        group_avgs[gname] = sum(np.nan_to_num(pairwise_avg[m], nan=0.0) * counts[m] for m in valid) / gc  # type: ignore[assignment]
        group_pres[gname] = sum(presence[m] * counts[m] for m in valid) / gc  # type: ignore[assignment]
        group_counts[gname] = gc

    for gname, avg in group_avgs.items():
        plot_heatmap_grid(
            avg,
            object_labels,
            object_boundaries,
            f"Pairwise Features: {gname} (n={group_counts[gname]:,})",
            save_path,
            f"pairwise_group_{gname}",
            feature_ranges=feature_ranges,
        )
        plot_presence_heatmap(
            group_pres[gname],
            object_labels,
            object_boundaries,
            f"Pair Presence: {gname} (n={group_counts[gname]:,})",
            save_path,
            f"pairwise_presence_group_{gname}",
        )

    # Group differences
    gnames = list(group_avgs.keys())
    for i, g1 in enumerate(gnames):
        for g2 in gnames[i + 1 :]:
            with np.errstate(invalid="ignore", divide="ignore"):
                diff = 100.0 * (group_avgs[g1] - group_avgs[g2]) / group_avgs[g2]

            plot_heatmap_grid(
                diff,
                object_labels,
                object_boundaries,
                f"Pairwise Δ%: {g1} vs {g2}",
                save_path,
                f"pairwise_diffpct_{g1}_vs_{g2}",
                cmap="RdBu_r",
                symmetric=True,
            )
            plot_presence_heatmap(
                group_pres[g1] - group_pres[g2],
                object_labels,
                object_boundaries,
                f"Pair Presence Δpp: {g1} vs {g2}",
                save_path,
                f"pairwise_presence_diff_{g1}_vs_{g2}",
                cmap="RdBu_r",
                symmetric=True,
            )


def plot_all_heatmaps(
    pairwise_avg: dict[str, np.ndarray],
    presence: dict[str, np.ndarray],
    counts: dict[str, int],
    object_labels: list[str],
    object_boundaries: list[int],
    save_path: str,
    custom_groups: dict[str, list[str]] | None = None,
) -> None:
    """Generate all heatmap plots."""
    class_names = _get_valid_classes(counts, pairwise_avg)
    if not class_names:
        logging.warning("No classes with data for heatmap plotting.")
        return

    feature_ranges = _compute_feature_ranges(pairwise_avg, class_names)

    _plot_class_heatmaps(
        pairwise_avg,
        presence,
        counts,
        class_names,
        object_labels,
        object_boundaries,
        save_path,
        feature_ranges,
    )
    _plot_difference_heatmaps(
        pairwise_avg,
        presence,
        counts,
        class_names,
        object_labels,
        object_boundaries,
        save_path,
    )

    if custom_groups:
        _plot_group_heatmaps(
            pairwise_avg,
            presence,
            counts,
            custom_groups,
            object_labels,
            object_boundaries,
            save_path,
            feature_ranges,
        )


def plot_attention_gradients(
    mean_grads: dict[str, np.ndarray],
    counts: dict[str, int],
    object_labels: list[str],
    object_boundaries: list[int],
    save_path: str,
    quantities: tuple[str, ...] = PAIRWISE_QUANTITIES,
) -> None:
    """Plot attention gradient heatmaps showing feature importance."""
    class_names = _get_valid_classes(counts, mean_grads)
    if not class_names:
        logging.warning("No gradient data available for plotting.")
        return

    for c in class_names:
        grad_masked = _mask_diagonal(mean_grads[c])

        plot_heatmap_grid(
            np.abs(grad_masked),
            object_labels,
            object_boundaries,
            f"Attention Gradient |∂A/∂f|: {c} (n={counts[c]:,})",
            save_path,
            f"attention_grad_abs_{c}",
            quantities=quantities,
            cmap="magma",
        )
        plot_heatmap_grid(
            grad_masked,
            object_labels,
            object_boundaries,
            f"Attention Gradient ∂A/∂f: {c} (n={counts[c]:,})",
            save_path,
            f"attention_grad_signed_{c}",
            quantities=quantities,
            cmap="RdBu_r",
            symmetric=True,
        )

    if len(class_names) > 1:
        total = sum(counts[c] for c in class_names)
        overall = sum(np.abs(_mask_diagonal(mean_grads[c])) * counts[c] for c in class_names) / total

        plot_heatmap_grid(
            overall,
            object_labels,
            object_boundaries,
            f"Attention Gradient |∂A/∂f|: All Classes (n={total:,})",
            save_path,
            "attention_grad_abs_overall",
            quantities=quantities,
            cmap="magma",
        )

    # Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = _get_colors(len(class_names))
    x = np.arange(len(quantities))
    width = 0.8 / len(class_names)

    for i, (c, color) in enumerate(zip(class_names, colors)):
        importance = np.nansum(np.abs(_mask_diagonal(mean_grads[c])), axis=(0, 1))
        ax.bar(x + i * width - 0.4 + width / 2, importance, width, label=c, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([QUANTITY_LATEX.get(q, q) for q in quantities], fontsize=11)
    ax.set_ylabel("Summed |∂Attention/∂Feature|", fontsize=11)
    ax.set_title("Pairwise Feature Importance for Attention", fontsize=14)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_plot(fig, f"{save_path}/attention_grad_feature_importance.pdf", use_format="pdf")


def _find_kinematic_indices(columns: list[str], short_name: str) -> dict[str, int | None]:
    """Find pt, eta, phi, e column indices for a given object type."""
    indices: dict[str, int | None] = {"pt": None, "eta": None, "phi": None, "e": None}
    for i, col in enumerate(columns):
        var_name = col.split(f"{short_name}_")[-1]
        if var_name in indices:
            indices[var_name] = i
    return indices


def build_object_specs(selection) -> list[ObjectSpec]:
    """Build object specifications from dataloader selection."""
    specs = []

    for obj_name in selection.keys():
        if obj_name == "events":
            continue

        short_name = OBJECTS_SHORT_NAMES.get(obj_name, obj_name[:3])
        columns = list(selection[obj_name].offset_used_columns)
        idx = _find_kinematic_indices(columns, short_name)

        if idx["pt"] is None or idx["eta"] is None or idx["phi"] is None:
            logging.warning(f"Skipping {obj_name}: missing pt/eta/phi columns")
            continue

        specs.append(
            ObjectSpec(
                name=obj_name,
                n_objects=selection[obj_name].n_objects,
                pad_value=selection[obj_name].pad_value,
                rest_mass=DEFAULT_PARTICLE_MASSES.get(short_name),
                pt_idx=idx["pt"],
                eta_idx=idx["eta"],
                phi_idx=idx["phi"],
                e_idx=idx["e"],
            )
        )

    return specs


def build_object_labels(specs: list[ObjectSpec]) -> tuple[list[str], list[int], int]:
    """Build object labels and boundaries from specs."""
    labels: list[str] = []
    boundaries: list[int] = []
    cumulative = 0

    for spec in specs:
        prefix = OBJECT_LABEL_PREFIX.get(spec.name, spec.name[0])
        labels.extend(f"{prefix}{i + 1}" for i in range(spec.n_objects))
        cumulative += spec.n_objects
        boundaries.append(cumulative)

    return labels, boundaries, cumulative


def extract_kinematics(
    batch_data: dict,
    specs: list[ObjectSpec],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract pt, eta, phi, energy, mask from batch for given indices."""
    pt_parts, eta_parts, phi_parts, energy_parts, mask_parts = [], [], [], [], []

    for spec in specs:
        X_obj = batch_data[spec.name][0][indices]
        pt = X_obj[:, :, spec.pt_idx]
        eta = X_obj[:, :, spec.eta_idx]
        phi = X_obj[:, :, spec.phi_idx]

        if spec.pad_value is not None:
            obj_mask = pt == spec.pad_value
        else:
            obj_mask = torch.zeros_like(pt, dtype=torch.bool)

        pt = pt.masked_fill(obj_mask, 0.0)
        eta = eta.masked_fill(obj_mask, 0.0)
        phi = phi.masked_fill(obj_mask, 0.0)

        if spec.e_idx is not None:
            energy = X_obj[:, :, spec.e_idx]
        else:
            energy = derive_energy_from_mass(pt, eta, spec.rest_mass or 0.0)
        energy = energy.masked_fill(obj_mask, 0.0)

        pt_parts.append(pt)
        eta_parts.append(eta)
        phi_parts.append(phi)
        energy_parts.append(energy)
        mask_parts.append(obj_mask)

    return (
        torch.cat(pt_parts, dim=1),
        torch.cat(eta_parts, dim=1),
        torch.cat(phi_parts, dim=1),
        torch.cat(energy_parts, dim=1),
        torch.cat(mask_parts, dim=1),
    )


def compute_attention_gradients(
    model: BaseLightningModule,
    dataloader,
    specs: list[ObjectSpec],
    class_labels: dict[str, int],
    n_events_per_class: int,
    device: torch.device,
) -> GradientAccumulator:
    """Compute gradients of model output w.r.t. pairwise features."""

    model.eval()
    accumulator = GradientAccumulator()
    label_to_name = {v: k for k, v in class_labels.items()}

    if not hasattr(model, "model") or not hasattr(model.model, "pairwise_calculator"):
        logging.error("Model does not have pairwise attention. Cannot compute gradients.")
        return accumulator

    pbar = tqdm(desc="Computing attention gradients", leave=False)

    for batch in dataloader:
        batch_data, _ = batch
        y_true = batch_data["events"][1]

        if y_true is None:
            continue

        y_np = y_true.cpu().numpy()
        batch_size = len(y_np)

        if accumulator.is_done(class_labels, n_events_per_class):
            break

        need_more = any(
            accumulator.get_count(label_to_name.get(int(idx), f"class_{idx}")) < n_events_per_class
            for idx in np.unique(y_np)
        )

        if not need_more:
            pbar.update(batch_size)
            continue

        X_events = batch_data["events"][0].to(device)
        Xs = tuple(batch_data[spec.name][0].to(device) for spec in specs)
        y_true = y_true.to(device)

        part_model: ParticleTransformer = model.model  # type: ignore[assignment]

        x_jagged, x_jagged_valid = part_model.jagged_preprocessor(*Xs)

        # Build pairwise features manually for gradient tracking
        pt_parts, eta_parts, phi_parts, energy_parts, mask_parts = [], [], [], [], []

        for spec_model in part_model._particle_attention_specs:
            tensor_idx = part_model.object_name_to_tensor_idx[spec_model.object_name]
            x_obj = Xs[tensor_idx]
            mask_slice = part_model.object_slices[spec_model.object_name]
            obj_mask = x_jagged_valid[:, mask_slice]

            pt = x_obj[:, :, spec_model.pt_index].masked_fill(obj_mask, 0.0)
            eta = x_obj[:, :, spec_model.eta_index].masked_fill(obj_mask, 0.0)
            phi = x_obj[:, :, spec_model.phi_index].masked_fill(obj_mask, 0.0)

            if spec_model.energy_index is not None:
                energy = x_obj[:, :, spec_model.energy_index]
            else:
                energy = derive_energy_from_mass(pt, eta, spec_model.rest_mass)  # type: ignore[arg-type]
            energy = energy.masked_fill(obj_mask, 0.0)

            pt_parts.append(pt)
            eta_parts.append(eta)
            phi_parts.append(phi)
            energy_parts.append(energy)
            mask_parts.append(obj_mask)

        pt_all = torch.cat(pt_parts, dim=1)
        eta_all = torch.cat(eta_parts, dim=1)
        phi_all = torch.cat(phi_parts, dim=1)
        energy_all = torch.cat(energy_parts, dim=1)
        mask_all = torch.cat(mask_parts, dim=1)

        pairwise_raw, pairwise_mask = part_model.pairwise_calculator(pt_all, eta_all, phi_all, energy_all, mask_all)
        pairwise_raw = pairwise_raw.detach().requires_grad_(True)

        pairwise_bias = part_model.pairwise_embedder(pairwise_raw, pairwise_mask.unsqueeze(1))
        particle_adjacency_mask = build_adjacency_attention_mask(x_jagged_valid)

        if part_model.flat_embeddings is not None:
            x_flat = part_model.flat_embeddings(X_events)
            x_jagged = part_model.embeddings_fuser(
                x_flat,
                x_jagged,
                mask=None if part_model.disable_flat_embeddings_mask else x_jagged_valid,
            )

        for block in part_model.particle_blocks:
            x_jagged = block(x_jagged, mask=particle_adjacency_mask, bias=pairwise_bias)

        x_jagged = part_model.final_particle_ln(x_jagged)

        cls_token = part_model.cls_token.expand(x_jagged.shape[0], -1, -1)
        cls_mask = torch.zeros(x_jagged.shape[0], 1, dtype=torch.bool, device=device)
        cls_mask = torch.cat([cls_mask, x_jagged_valid], dim=1)
        cls_mask = rearrange(cls_mask, "b n -> b 1 1 n")

        for block in part_model.class_blocks:
            cls_token = block(cls_token, x_jagged, mask=cls_mask)

        logits = part_model.to_logits(cls_token.squeeze(1))
        logits.abs().sum().backward()

        if pairwise_raw.grad is not None:
            grads = pairwise_raw.grad.detach().cpu().numpy()

            for label_idx in np.unique(y_np):
                class_name = label_to_name.get(int(label_idx), f"class_{label_idx}")
                if accumulator.get_count(class_name) >= n_events_per_class:
                    continue

                cls_mask_np = y_np == label_idx
                n_to_add = min(
                    int(cls_mask_np.sum()),
                    n_events_per_class - accumulator.get_count(class_name),
                )

                if n_to_add > 0:
                    sel = np.flatnonzero(cls_mask_np)[:n_to_add]
                    accumulator.add_batch(class_name, grads[sel].mean(axis=0), n_to_add)

            pairwise_raw.grad = None

        pbar.update(batch_size)

    pbar.close()
    return accumulator


def _process_batch(
    batch_data: dict,
    specs: list[ObjectSpec],
    calculator: PairwiseFeaturesCalculator,
    accumulator: PairwiseAccumulator,
    class_labels: dict[str, int],
    n_events_per_class: int,
) -> bool:
    """Process a single batch. Returns True if done collecting."""
    y_true = batch_data["events"][1]
    if y_true is None:
        return False

    y_np = y_true.cpu().numpy()
    label_to_name = {v: k for k, v in class_labels.items()}

    # Select events we still need
    keep_indices = []
    for label_idx in np.unique(y_np):
        class_name = label_to_name.get(int(label_idx), f"class_{label_idx}")
        remaining = n_events_per_class - accumulator.get_count(class_name)
        if remaining > 0:
            idxs = np.flatnonzero(y_np == label_idx)
            keep_indices.extend(idxs[: int(remaining)].tolist())

    if not keep_indices:
        return accumulator.is_done(class_labels, n_events_per_class)

    keep_indices.sort()
    keep_t = torch.as_tensor(keep_indices, dtype=torch.long, device=y_true.device)
    y_sel = y_true[keep_t].cpu().numpy()

    pt, eta, phi, energy, mask = extract_kinematics(batch_data, specs, keep_t)

    with torch.no_grad():
        pairwise_features, _ = calculator(pt, eta, phi, energy, mask)

    # Set diagonal (i=j) to NaN - these are zero by construction
    n_obj = pairwise_features.shape[1]
    diag_idx = torch.arange(n_obj, device=pairwise_features.device)
    pairwise_features[:, diag_idx, diag_idx, :] = float("nan")

    # Valid pairs mask: exclude padded particles
    valid_pairs = (~mask).unsqueeze(2) & (~mask).unsqueeze(1)

    # Accumulate per class
    for label_idx in np.unique(y_sel):
        class_name = label_to_name.get(int(label_idx), f"class_{label_idx}")
        if accumulator.get_count(class_name) >= n_events_per_class:
            continue

        cls_mask = y_sel == label_idx
        n_to_add = min(int(cls_mask.sum()), n_events_per_class - accumulator.get_count(class_name))

        if n_to_add > 0:
            sel = np.flatnonzero(cls_mask)[:n_to_add]
            accumulator.add_batch(class_name, pairwise_features[sel], valid_pairs[sel], n_to_add)

    return accumulator.is_done(class_labels, n_events_per_class)


def run_plot_pairwise(
    files: str | list[str],
    column_names: list[str],
    n_events_per_class: int = 100_000,
    dataset_kwargs: dict | None = None,
    dataloader_kwargs: dict | None = None,
    custom_groups: dict[str, list[str]] | None = None,
    class_labels: dict[str, int] | None = None,
    model_checkpoint: str | None = None,
    config: DictConfig | None = None,
) -> None:
    """Compute and plot pairwise features from HDF5 data."""
    dataset_kwargs = dataset_kwargs or {}
    dataloader_kwargs = dataloader_kwargs or {}

    logging.info("Initializing dataloader...")
    dataloader, selection, num_entries = get_ml_hdf5_dataloader(
        name="pairwisePlotter",
        files=files,
        column_names=column_names,
        shuffle=True,
        collate_fn=full_collate_fn,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    specs = build_object_specs(selection)
    if not specs:
        logging.error("No valid object types with pt/eta/phi found.")
        return

    logging.info(f"Object types: {[s.name for s in specs]}")

    object_labels, object_boundaries, total_objects = build_object_labels(specs)
    logging.info(f"Total objects per event: {total_objects}")

    calculator = PairwiseFeaturesCalculator(quantities=None)
    accumulator = PairwiseAccumulator()

    # Get class labels from selection if not provided
    if class_labels is None:
        try:
            class_labels = dict(selection["events"].labels)
        except Exception:
            logging.error("No class labels available.")
            return

    pbar = tqdm(desc="Processing events", total=num_entries, leave=False)

    for batch in dataloader:
        batch_data, _ = batch
        batch_size = batch_data["events"][0].shape[0]

        done = _process_batch(batch_data, specs, calculator, accumulator, class_labels, n_events_per_class)

        pbar.update(batch_size)

        if done:
            logging.info(f"Reached {n_events_per_class} events per class.")
            break

    pbar.close()

    pairwise_avg, presence, samples, counts = accumulator.finalize()

    if not pairwise_avg:
        logging.error("No pairwise features accumulated.")
        return

    for name in class_labels:
        counts.setdefault(name, 0)

    logging.info(f"Events per class: {counts}")

    missing = [k for k, v in counts.items() if v == 0]
    if missing:
        logging.warning(f"No data for classes: {', '.join(missing)}")

    save_path = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "pairwise")
    os.makedirs(save_path, exist_ok=True)

    hep.style.use(hep.style.ATLAS)
    logging.info(f"Generating plots in {save_path}...")

    plot_all_heatmaps(
        pairwise_avg,
        presence,
        counts,
        object_labels,
        object_boundaries,
        save_path,
        custom_groups,
    )
    plot_1d_distributions(samples, counts, save_path)
    plot_2d_correlations(samples, counts, save_path)
    plot_2d_density(samples, counts, save_path)

    if model_checkpoint is not None and config is not None:
        _run_attention_gradient_analysis(
            model_checkpoint,
            config,
            files,
            column_names,
            specs,
            object_labels,
            object_boundaries,
            class_labels,
            min(n_events_per_class, 10_000),
            save_path,
            dataset_kwargs,
            dataloader_kwargs,
        )

    logging.info("Pairwise feature plotting complete!")


def _run_attention_gradient_analysis(
    model_checkpoint: str,
    config: DictConfig,
    files: str | list[str],
    column_names: list[str],
    specs: list[ObjectSpec],
    object_labels: list[str],
    object_boundaries: list[int],
    class_labels: dict[str, int],
    n_events_per_class: int,
    save_path: str,
    dataset_kwargs: dict | None,
    dataloader_kwargs: dict | None,
) -> None:
    """Run attention gradient analysis using a trained model."""

    logging.info("Running attention gradient analysis...")

    try:
        config = copy.deepcopy(config)
        with open_dict(config):
            config.model_config.load_checkpoint = os.path.basename(model_checkpoint)
            config.model_config.training_config.model_save_path = os.path.dirname(model_checkpoint)

        logging.info(f"Loading model from {model_checkpoint}...")
        model, _ = load_sig_bkg_model(config, events_only=False, checkpoint_path=model_checkpoint)

        accelerator = config.experiment_config.accelerator
        device = torch.device("cuda" if accelerator == "gpu" and torch.cuda.is_available() else accelerator)
        model = model.to(device)
        model.eval()

        if not hasattr(model, "model") or not hasattr(model.model, "particle_attention"):
            logging.warning("Model does not have pairwise attention. Skipping.")
            return

        if model.model.particle_attention is None:
            logging.warning("Model's pairwise attention is not configured. Skipping.")  # type: ignore[unreachable]
            return

        feature_scaling = config.dataset_config.get("feature_scaling")
        grad_kwargs = (dataset_kwargs or {}).copy()
        if feature_scaling:
            grad_kwargs.update(
                {
                    "numer_scaler_type": feature_scaling.get("numer_scaler_type"),
                    "categ_scaler_type": feature_scaling.get("categ_scaler_type"),
                    "scaler_path": feature_scaling.get("save_path"),
                    "scalers_extra_hash": str(config.dataset_config.files),
                }
            )

        logging.info("Creating dataloader for gradient analysis...")
        grad_dataloader, _, _ = get_ml_hdf5_dataloader(
            name="gradientAnalysis",
            files=files,
            column_names=column_names,
            shuffle=True,
            collate_fn=full_collate_fn,
            dataset_kwargs=grad_kwargs,
            dataloader_kwargs=dataloader_kwargs or {},
        )

        grad_accumulator = compute_attention_gradients(
            model, grad_dataloader, specs, class_labels, n_events_per_class, device
        )

        mean_grads, _, grad_counts = grad_accumulator.finalize()

        if not mean_grads:
            logging.warning("No gradients computed.")
            return

        logging.info(f"Gradient analysis events per class: {grad_counts}")

        hep.style.use(hep.style.ATLAS)
        plot_attention_gradients(
            mean_grads,
            grad_counts,
            object_labels,
            object_boundaries,
            save_path,
        )

        logging.info("Attention gradient analysis complete!")

    except Exception as e:
        logging.error(f"Attention gradient analysis failed: {e}")

        traceback.print_exc()


def _resolve_config(config_arg: str) -> tuple[str, list[str] | None]:
    """Resolve CLI config argument to (config_dir, overrides)."""
    base = os.environ["ANALYSIS_ML_CONFIG_DIR"]
    cfg_dir = os.path.join(base, config_arg)

    def has_config(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "training_config.yaml"))

    if os.path.isdir(cfg_dir) and has_config(cfg_dir):
        return cfg_dir, None

    if "/" not in config_arg:
        raise RuntimeError(f"Config not found: {cfg_dir}")

    cfg_name, dataset_token = config_arg.split("/", 1)
    cfg_dir = os.path.join(base, cfg_name)

    if not (os.path.isdir(cfg_dir) and has_config(cfg_dir)):
        raise RuntimeError(f"Config not found: {cfg_dir}")

    ds_dir = os.path.join(cfg_dir, "dataset_config")
    if not os.path.isdir(ds_dir):
        return cfg_dir, None

    # Try exact match
    if os.path.isfile(os.path.join(ds_dir, f"{dataset_token}.yaml")):
        return cfg_dir, [f"dataset_config={dataset_token}"]

    # Try shorthand
    if dataset_token in DATASET_SHORTHAND:
        shorthand = DATASET_SHORTHAND[dataset_token]
        if os.path.isfile(os.path.join(ds_dir, f"{shorthand}.yaml")):
            return cfg_dir, [f"dataset_config={shorthand}"]

    # Fuzzy match
    try:
        available = [f[:-5] for f in os.listdir(ds_dir) if f.endswith(".yaml")]
    except OSError:
        return cfg_dir, None

    matches = [n for n in available if dataset_token in n]
    if not matches:
        return cfg_dir, None

    prefs = ["jagged", "flat"] if dataset_token == "2L" else ["flat", "jagged"]
    for p in prefs:
        hit = next((m for m in matches if p in m), None)
        if hit:
            return cfg_dir, [f"dataset_config={hit}"]

    return cfg_dir, [f"dataset_config={matches[0]}"]


def _infer_default_config() -> str:
    """Infer default config from ANALYSIS_ML_DATA_DIR."""
    data_dir = os.environ.get("ANALYSIS_ML_DATA_DIR", "")
    base = os.path.basename(os.path.normpath(data_dir)) if data_dir else ""

    for ch in ("0L", "1L", "2L"):
        if ch in base:
            return f"signal/{ch}"

    return "signal"


def _parse_class_labels(config: DictConfig, labels: dict | None) -> tuple[dict[str, int], dict]:
    """Parse class labels from config."""

    classes = config.dataset_config.get("classes")
    dataset_kwargs: dict = {}

    if classes is None:
        if labels is not None:
            return dict(labels), {"class_labels": dict(labels)}
        return {}, {}

    class_labels = {}
    for i, config_key in enumerate(classes):
        if isinstance(config_key, str):
            class_labels[config_key] = i
        elif isinstance(config_key, OmegaDictConfig):
            if len(config_key.keys()) != 1:
                raise ValueError(f"Expected single key in class config: {config_key}")
            class_labels[str(list(config_key.keys())[0])] = i
        else:
            raise ValueError(f"Unsupported class key type: {type(config_key)}")

    if labels is not None:
        _, remap_labels, _ = get_classifier_labels(classes, labels)
        dataset_kwargs["remap_labels"] = remap_labels
        dataset_kwargs["max_label"] = max(labels.values())

    dataset_kwargs["class_labels"] = class_labels
    return class_labels, dataset_kwargs


def _get_custom_groups(config: DictConfig) -> dict[str, list[str]] | None:
    """Extract custom groups from config."""
    for section in ("dataset_config", "plotting_config"):
        cfg = getattr(config, section, None)
        if cfg is None:
            continue

        for key in ("custom_groups", "scores"):
            if key in cfg:
                return {k: list(v) for k, v in dict(cfg[key]).items()}

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ParT-style pairwise features from HDF5 data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    plot_pairwise
    plot_pairwise 50000
    plot_pairwise -c signal/2L
    plot_pairwise -c signal/2L -n 10000
    plot_pairwise -c signal/2L -m /path/to/model.ckpt
        """,
    )
    parser.add_argument("n_events_positional", nargs="?", type=int, help="Max events per class")
    parser.add_argument("-c", "--config", type=str, help="Config path (e.g., signal/2L)")
    parser.add_argument("-n", "--n_events", type=int, default=100_000, help="Max events per class")
    parser.add_argument("-s", "--scale", action="store_true", help="Use scaled features")
    parser.add_argument("-m", "--model", type=str, help="Model checkpoint for gradient analysis")

    args = parser.parse_args()

    if args.n_events_positional is not None:
        args.n_events = args.n_events_positional

    config_arg = args.config or _infer_default_config()
    cfg_dir, overrides = _resolve_config(config_arg)

    config = get_hydra_config(cfg_dir, "training_config", overrides=overrides)
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    dataset_kwargs = dict(config.dataset_config.dataset_kwargs)

    if args.scale:
        logging.info("Using feature scaling.")
        fs = config.dataset_config.feature_scaling
        dataset_kwargs.update(
            {
                "numer_scaler_type": fs.get("numer_scaler_type"),
                "categ_scaler_type": fs.get("categ_scaler_type"),
                "scaler_path": fs.get("save_path"),
                "scalers_extra_hash": str(config.dataset_config.files),
            }
        )

    metadata = get_hdf5_metadata(config.dataset_config.files, resolve_path=True)
    labels = metadata.get("labels")

    class_labels, label_kwargs = _parse_class_labels(config, labels)
    dataset_kwargs.update(label_kwargs)

    custom_groups = _get_custom_groups(config)

    logging.info(f"Processing up to {args.n_events} events per class...")

    run_plot_pairwise(
        files=config.dataset_config.files,
        column_names=config.dataset_config.features,
        n_events_per_class=args.n_events,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dict(config.dataset_config.dataloader_kwargs),
        custom_groups=custom_groups,
        class_labels=class_labels or None,
        model_checkpoint=args.model,
        config=config,
    )


if __name__ == "__main__":
    main()
