import glob
import logging
import os
from dataclasses import dataclass

import h5py
import hist
import hydra
import matplotlib.pyplot as plt
import numpy as np
from f9columnar.utils.helpers import hist_to_unumpy
from omegaconf import DictConfig, ListConfig
from plothist import plot_data_model_comparison, plot_error_hist, plot_hist
from uncertainties import unumpy

from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.loggers import log_hydra_config, setup_logger
from seesawml.utils.plots_utils import atlas_label, get_color, save_plot


@dataclass(frozen=True, slots=True, eq=False)
class MCVariation:
    name: str
    nom_h: hist.Hist
    variation_val: float | None = None
    up_h: hist.Hist | None = None
    down_h: hist.Hist | None = None

    @property
    def nom(self) -> hist.Hist:
        return self.nom_h

    @property
    def up(self) -> hist.Hist:
        if self.up_h is None:
            raise ValueError("Up variation not set.")
        return self.up_h

    @property
    def down(self) -> hist.Hist:
        if self.down_h is None:
            raise ValueError("Down variation not set.")
        return self.down_h

    @property
    def variation(self) -> float | None:
        return self.variation_val

    def __str__(self):
        return f"MCVariation(name={self.name}, variation={self.variation}, nbins={len(self.nom_h.axes[0])})"

    def __repr__(self):
        return self.__str__()

    def get_variation(self, variation: str) -> hist.Hist:
        if variation == "nominal" or variation == "nom":
            return self.nom
        elif variation == "up":
            return self.up
        elif variation == "down":
            return self.down
        else:
            raise ValueError(f"Invalid variation: {variation}. Choose from 'nominal', 'up', 'down'.")

    def has_variation(self, variation: str) -> bool:
        if variation == "nominal" or variation == "nom":
            return True
        elif variation == "up":
            return self.up_h is not None
        elif variation == "down":
            return self.down_h is not None
        else:
            raise ValueError(f"Invalid variation: {variation}. Choose from 'nominal', 'up', 'down'.")


class MCVariationMaker:
    def __init__(self, bins: np.ndarray) -> None:
        self.bins = bins

    def make(self, name: str, nominal: np.ndarray, weights: np.ndarray, variation: float | None = None) -> MCVariation:
        nom_h = hist.Hist(hist.axis.Variable(self.bins), storage=hist.storage.Weight())
        nom_h.fill(nominal, weight=weights)

        if variation is None:
            return MCVariation(name, nom_h=nom_h, variation_val=None)

        up_weights = weights * (1.0 + variation)
        down_weights = weights * (1.0 - variation)

        up_h = hist.Hist(hist.axis.Variable(self.bins), storage=hist.storage.Weight())
        up_h.fill(nominal, weight=up_weights)

        down_h = hist.Hist(hist.axis.Variable(self.bins), storage=hist.storage.Weight())
        down_h.fill(nominal, weight=down_weights)

        return MCVariation(name, nom_h=nom_h, variation_val=variation, up_h=up_h, down_h=down_h)


def load_saved_dataset(save_path: str) -> tuple[np.ndarray, list[str]]:
    hdf5_files = glob.glob(f"{save_path}.hdf5")

    piles = []

    if len(hdf5_files) > 1:
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, "r") as f:
                column_names = f["events"].dtype.names
                pile_dataset = np.stack([f["events"][col][:] for col in column_names], axis=1)
            piles.append(pile_dataset)
    else:
        with h5py.File(hdf5_files[0], "r") as f:
            groups = f["events"].keys()
            column_names = f["events"]["p0"].dtype.names

            for g in groups:
                pile_dataset = np.stack([f["events"][g][col][:] for col in column_names], axis=1)
                piles.append(pile_dataset)

    dataset = np.concatenate(piles, axis=0)

    if dataset[np.all(dataset == 0, axis=1)].shape[0] > 0:
        logging.warning("Dataset contains empty rows, which is not allowed!")

    logging.info(f"Loaded dataset with {dataset.shape[0]} events and {dataset.shape[1]} columns.")

    return dataset, column_names


def split_dataset(dataset: np.ndarray, column_names: list[str]) -> dict[str, np.ndarray]:
    data_label, mc_label = 1, 0
    loose_label, tight_label = 2, 0

    data_type_idx = column_names.index("data_type")
    toy_type_idx = column_names.index("toy_type")

    data_type_col = dataset[:, data_type_idx]
    toy_type_col = dataset[:, toy_type_idx]

    dataset = np.delete(dataset, [data_type_idx, toy_type_idx], axis=1)

    loose_data = dataset[(data_type_col == data_label) & (toy_type_col == loose_label)]
    tight_data = dataset[(data_type_col == data_label) & (toy_type_col == tight_label)]

    loose_mc = dataset[(data_type_col == mc_label) & (toy_type_col == loose_label)]
    tight_mc = dataset[(data_type_col == mc_label) & (toy_type_col == tight_label)]

    return {"loose_data": loose_data, "tight_data": tight_data, "loose_mc": loose_mc, "tight_mc": tight_mc}


def get_dataset_hists(
    data_dct: dict[str, np.ndarray],
    pt_min: float,
    pt_max: float,
    nbins: int | list[float],
    variation: float | None = 0.1,
) -> tuple[np.ndarray, dict[str, MCVariation]]:
    loose_data, tight_data, loose_mc, tight_mc = (
        data_dct["loose_data"],
        data_dct["tight_data"],
        data_dct["loose_mc"],
        data_dct["tight_mc"],
    )

    if isinstance(nbins, list) or isinstance(nbins, ListConfig):  # type: ignore
        bins = np.array(list(nbins))
    else:
        bins = np.logspace(np.log(pt_min), np.log(pt_max), nbins, base=np.e)  # type: ignore

    var_maker = MCVariationMaker(bins)

    loose_data_var = var_maker.make("loose_data", loose_data[:, 0], loose_data[:, 1], None)
    tight_data_var = var_maker.make("tight_data", tight_data[:, 0], tight_data[:, 1], None)
    loose_mc_var = var_maker.make("loose_mc", loose_mc[:, 0], loose_mc[:, 1], variation)
    tight_mc_var = var_maker.make("tight_mc", tight_mc[:, 0], tight_mc[:, 1], variation)

    return bins, {
        "loose_data": loose_data_var,
        "tight_data": tight_data_var,
        "loose_mc": loose_mc_var,
        "tight_mc": tight_mc_var,
    }


def get_binned_fake_factor(
    variation_dct: dict[str, MCVariation],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ff_variations_nom, ff_variations_std = {}, {}

    for v in ["nominal", "up", "down"]:
        loose_data_h, tight_data_h, loose_mc_h, tight_mc_h = (
            variation_dct["loose_data"].get_variation("nom"),
            variation_dct["tight_data"].get_variation("nom"),
            variation_dct["loose_mc"].get_variation(v),
            variation_dct["tight_mc"].get_variation(v),
        )
        loose_data_unp = hist_to_unumpy(loose_data_h, zero_to_nan=False, flatten=True)
        tight_data_unp = hist_to_unumpy(tight_data_h, zero_to_nan=False, flatten=True)

        loose_mc_unp = hist_to_unumpy(loose_mc_h, zero_to_nan=False, flatten=True)
        tight_mc_unp = hist_to_unumpy(tight_mc_h, zero_to_nan=False, flatten=True)

        num = tight_data_unp - tight_mc_unp
        denom = loose_data_unp - loose_mc_unp

        ff = np.zeros_like(num)
        mask = unumpy.nominal_values(denom) != 0.0
        ff[mask] = num[mask] / denom[mask]

        ff_nom, ff_std = unumpy.nominal_values(ff), unumpy.std_devs(ff)

        ff_variations_nom[v] = ff_nom
        ff_variations_std[v] = ff_std

    return ff_variations_nom, ff_variations_std


def plot_saved_dataset(
    dataset_dct: dict[str, np.ndarray],
    variation_dct: dict[str, MCVariation],
    weight_bins: int,
    save_path: str,
    atlas_marker: str | None = None,
) -> None:
    loose_data, tight_data, loose_mc, tight_mc = (
        dataset_dct["loose_data"],
        dataset_dct["tight_data"],
        dataset_dct["loose_mc"],
        dataset_dct["tight_mc"],
    )
    loose_data_h, tight_data_h, loose_mc_h, tight_mc_h = (
        variation_dct["loose_data"].get_variation("nom"),
        variation_dct["tight_data"].get_variation("nom"),
        variation_dct["loose_mc"].get_variation("nom"),
        variation_dct["tight_mc"].get_variation("nom"),
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    plot_error_hist(loose_data_h, ax=axs[0], color="black", label="data loose")
    plot_hist(loose_mc_h, ax=axs[0], histtype="step", color="C0", linewidth=2, label="MC loose")

    plot_error_hist(tight_data_h, ax=axs[1], color="black", label="data tight")
    plot_hist(tight_mc_h, ax=axs[1], histtype="step", color="C0", linewidth=2, label="MC tight")

    for ax in axs:
        ax.set_ylabel("Events")
        ax.set_xlabel(r"toy $p_{\mathrm{T}}$")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()

    if atlas_marker is not None:
        atlas_label(axs[0], llabel=atlas_marker, loc=0, fontsize=12)

    save_plot(fig, f"{save_path}/toy_dataset.pdf")

    data_weights = np.concatenate([loose_data[:, 1], tight_data[:, 1]])
    mc_weights = np.concatenate([loose_mc[:, 1], tight_mc[:, 1]])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].hist(data_weights, bins=weight_bins, histtype="step", color="C0", lw=1.8)
    axs[0].set_xlabel("Toy data weights")

    axs[1].hist(mc_weights, bins=weight_bins, histtype="step", color="C0", lw=1.8)
    axs[1].set_xlabel("Toy MC weights")

    for ax in axs:
        ax.set_ylabel("Events")
        ax.set_yscale("log")

    if atlas_marker is not None:
        atlas_label(axs[0], llabel=atlas_marker, loc=0, fontsize=12)

    save_plot(fig, f"{save_path}/toy_dataset_weights.pdf")


def construct_up_down_error_band(
    nom: np.ndarray, err_nom: np.ndarray, up: np.ndarray, err_up: np.ndarray, down: np.ndarray, err_down: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta_up = up - nom
    delta_down = nom - down

    err_up_tot = np.sqrt(err_nom**2 + delta_up**2 + err_up**2)
    err_down_tot = np.sqrt(err_nom**2 + delta_down**2 + err_down**2)

    return err_up_tot, err_down_tot


def plot_binned_fake_factor(
    variation_dct: dict[str, MCVariation],
    ff_nom: dict[str, np.ndarray],
    ff_std: dict[str, np.ndarray],
    save_path: str,
    atlas_marker: str | None = None,
    variation: float | None = None,
) -> None:
    loose_data_h = variation_dct["loose_data"].get_variation("nom")

    ff_nom, ff_std = get_binned_fake_factor(variation_dct)

    fig, ax = plt.subplots(figsize=(7, 6))

    centers = loose_data_h.axes[0].centers
    widths = loose_data_h.axes[0].widths / 2

    nom = ff_nom["nominal"]

    err_down, err_up = construct_up_down_error_band(
        nom,
        ff_std["nominal"],
        ff_nom["up"],
        ff_std["up"],
        ff_nom["down"],
        ff_std["down"],
    )

    if variation is not None:
        up_label = f"MC up ({variation * 100:.1f}%)"
        down_label = f"MC down ({variation * 100:.1f}%)"
    else:
        up_label = "MC up"
        down_label = "MC down"

    ax.scatter(centers, nom, label="Nominal", color="black", zorder=2)
    ax.scatter(centers, nom + err_up, label=up_label, color="black", zorder=2, marker="^")
    ax.scatter(centers, nom - err_down, label=down_label, color="black", zorder=2, marker="v")

    ax.fill_between(centers, nom - err_down, nom + err_up, color="yellow", alpha=0.7, label="Error band", zorder=1)

    ax.set_xlim(np.min(centers) - np.min(widths), np.max(centers) + np.max(widths))
    ax.set_xscale("log")

    ax.set_xlabel(r"toy $p_{\mathrm{T}}$")
    ax.set_ylabel("Toy fake factor")

    ax.legend()

    if atlas_marker is not None:
        atlas_label(ax, llabel=atlas_marker, loc=0, fontsize=12)

    save_plot(fig, f"{save_path}/toy_hist_fake_factor.pdf")


def plot_binned_fake_factor_closure(
    dataset_dct: dict[str, np.ndarray],
    variation_dct: dict[str, MCVariation],
    ff_nom: dict[str, np.ndarray],
    ff_std: dict[str, np.ndarray],
    ff_bins: np.ndarray,
    save_path: str,
    atlas_marker: str | None = None,
) -> None:
    loose_data, _, loose_mc, _ = (
        dataset_dct["loose_data"],
        dataset_dct["tight_data"],
        dataset_dct["loose_mc"],
        dataset_dct["tight_mc"],
    )
    loose_data_h, tight_data_h, _, tight_mc_h = (
        variation_dct["loose_data"].nom,
        variation_dct["tight_data"].nom,
        variation_dct["loose_mc"].nom,
        variation_dct["tight_mc"].nom,
    )

    nom = ff_nom["nominal"]

    err_down, err_up = construct_up_down_error_band(
        nom,
        ff_std["nominal"],
        ff_nom["up"],
        ff_std["up"],
        ff_nom["down"],
        ff_std["down"],
    )

    loose_mc, loose_mc_w = loose_mc[:, 0], loose_mc[:, 1]
    loose_data, loose_data_w = loose_data[:, 0], loose_data[:, 1]

    bins = loose_data_h.axes[0].edges

    loose_mc_bin_idx = np.clip(np.digitize(loose_mc, ff_bins) - 1, 0, len(ff_bins) - 2)
    loose_data_bin_idx = np.clip(np.digitize(loose_data, ff_bins) - 1, 0, len(ff_bins) - 2)

    loose_mc_bin_ff = ff_nom["nominal"][loose_mc_bin_idx]
    loose_data_bin_ff = ff_nom["nominal"][loose_data_bin_idx]

    loose_mc_ff_h = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
    loose_data_ff_h = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())

    loose_mc_ff_h.fill(loose_mc, weight=-loose_mc_w * loose_mc_bin_ff)
    loose_data_ff_h.fill(loose_data, weight=loose_data_w * loose_data_bin_ff)

    fake_counts = loose_data_ff_h + loose_mc_ff_h

    fig, ax, ax_comparison = plot_data_model_comparison(
        data_hist=tight_data_h,
        stacked_components=[tight_mc_h, fake_counts],
        stacked_labels=["MC", "Fakes"],
        stacked_colors=[get_color("Blue").rgb, get_color("Pink").rgb],
        xlabel=r"$\mathrm{toy}$ $p_{\mathrm{T}}$",
        ylabel=r"$\mathrm{Events}$",
        comparison="split_ratio",
        model_uncertainty=True,
        data_uncertainty_type="symmetrical",
    )
    fig.set_size_inches(6, 7.2)

    # histogram binning
    bin_edges = tight_data_h.axes[0].edges

    model_nom = (tight_mc_h.values() + fake_counts.values()).astype(float)

    # fake–factor bin centers
    ff_centers = 0.5 * (ff_bins[1:] + ff_bins[:-1])

    # interpolate errors on bin edges (not only centers)
    x_ext = np.concatenate(([bin_edges[0]], ff_centers, [bin_edges[-1]]))
    err_up_ext = np.concatenate(([err_up[0]], err_up, [err_up[-1]]))
    err_down_ext = np.concatenate(([err_down[0]], err_down, [err_down[-1]]))

    # interpolate onto all *bin edges* of the histogram
    err_up_interp = np.interp(bin_edges, x_ext, err_up_ext)
    err_down_interp = np.interp(bin_edges, x_ext, err_down_ext)

    # average uncertainties per bin
    err_up_per_bin = 0.5 * (err_up_interp[:-1] + err_up_interp[1:])
    err_down_per_bin = 0.5 * (err_down_interp[:-1] + err_down_interp[1:])

    # expand nominal model from centers to edges (step plot style)
    bin_edges_step = np.repeat(bin_edges, 2)[1:-1]

    # build error bands on edges
    model_up_edges = np.repeat(model_nom * (1 + err_up_per_bin), 2)
    model_down_edges = np.repeat(model_nom * (1 - err_down_per_bin), 2)

    # shade full area across all bins
    ax.fill_between(
        bin_edges_step,
        model_down_edges,
        model_up_edges,
        step="pre",
        color="gray",
        alpha=0.3,
        label="FF uncertainty",
    )

    # ratio panel band (relative uncertainties around 1)
    ratio_up_step = np.repeat(1 + err_up_per_bin, 2)
    ratio_down_step = np.repeat(1 - err_down_per_bin, 2)

    ax_comparison.fill_between(
        bin_edges_step,
        ratio_down_step,
        ratio_up_step,
        step="pre",
        color="gray",
        alpha=0.3,
        label="_unc_band",  # avoid duplicate legend entry
    )

    ax_comparison.set_ylim(0.5, 1.5)

    ax.legend()

    ax_comparison.set_xscale("log")

    ax.set_xscale("log")
    ax.set_yscale("log")

    if atlas_marker is not None:
        atlas_label(ax, llabel=atlas_marker, loc=0, fontsize=12)

    save_plot(fig, f"{save_path}/toy_hist_fake_factor_closure.pdf")


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig):
    setup_logger(config.min_logging_level)

    setup_analysis_dirs(config, verbose=False)
    log_hydra_config(config)

    toy_dataset_config = config.dataset_config
    plotting_config = config.plotting_config.toy_plot

    mc_variation = plotting_config.get("mc_variation", 0.1)
    logging.info(f"Using MC variation of {mc_variation * 100:.1f}% for fake factor estimation.")

    dataset, column_names = load_saved_dataset(toy_dataset_config.files)
    dataset_dct = split_dataset(dataset, column_names)

    pt_min, pt_max = plotting_config.pt_min, plotting_config.pt_max

    _, variation_dct = get_dataset_hists(dataset_dct, pt_min, pt_max, plotting_config.dataset_bins, mc_variation)
    ff_bins_arr, hists_ff_dct = get_dataset_hists(dataset_dct, pt_min, pt_max, plotting_config.ff_bins, mc_variation)

    ff_nom, ff_std = get_binned_fake_factor(hists_ff_dct)

    logging.info("Plotting toy dataset and fake factors.")

    plots_path = f"{os.environ['ANALYSIS_ML_RESULTS_DIR']}/fakes_toy_plots"
    os.makedirs(plots_path, exist_ok=True)

    atlas_marker = plotting_config.get("atlas_label", None)

    plot_saved_dataset(dataset_dct, variation_dct, plotting_config.weight_bins, plots_path, atlas_marker)
    plot_binned_fake_factor(hists_ff_dct, ff_nom, ff_std, plots_path, atlas_marker, mc_variation)
    plot_binned_fake_factor_closure(dataset_dct, variation_dct, ff_nom, ff_std, ff_bins_arr, plots_path, atlas_marker)


if __name__ == "__main__":
    main()
