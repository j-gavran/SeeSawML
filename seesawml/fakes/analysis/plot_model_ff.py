import itertools
import os

import hydra
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from f9columnar.utils.helpers import load_pickle
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig
from tqdm import tqdm
from uncertainties import unumpy as unp

from seesawml.fakes.analysis.plot_density_ratio import DensityRatioResult, get_density_ratio
from seesawml.fakes.training.tracker_plotting import atlas_label
from seesawml.utils.helpers import (
    setup_analysis_dirs,
)
from seesawml.utils.loggers import setup_logger


def get_latex_variable_name(variable: str) -> str:
    if variable == "pt":
        return r"$p_{\mathrm{T}}$"
    if variable == "abs_eta":
        return r"$|\eta|$"
    elif variable == "eta":
        return r"$\eta$"
    elif variable == "met":
        return r"$E_{\mathrm{T}}^{\mathrm{miss}}$"
    else:
        return variable


def get_binned_ff(
    variable: str, ff_averaged_bins: dict[str, np.ndarray], ff_combo: dict[str, tuple[float, float]], ff: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ff_order = {"pt": 0, "abs_eta": 1, "met": 2}

    ff_idx = {}

    for ff_name, ff_bounds in ff_combo.items():
        if ff_name == "eta":
            ff_name = "abs_eta"
        elif ff_name not in ff_averaged_bins:
            continue
        else:
            pass

        ff_bins = ff_averaged_bins[ff_name]
        edge_l, edge_r = ff_bounds
        take_binned_ff_at = (edge_l + edge_r) / 2

        idx = np.clip(np.digitize(take_binned_ff_at, ff_bins) - 1, 0, len(ff_bins) - 2)

        ff_idx[ff_name] = idx

    slice_idx = [slice(None)] * len(ff_order)
    for ff_name, idx in ff_idx.items():
        order = ff_order[ff_name]
        slice_idx[order] = idx  # type: ignore[call-overload]

    ff_binned = ff[tuple(slice_idx)]  # type: ignore[index]

    ff_binned_nom, ff_binned_std = unp.nominal_values(ff_binned), unp.std_devs(ff_binned)

    if variable == "eta":
        variable = "abs_eta"

    binned_x = ff_averaged_bins[variable]

    return binned_x, ff_binned_nom, ff_binned_std


def plot_integrated_ff(
    x_range: np.ndarray,
    ff_vals: np.ndarray,
    ff_errs: np.ndarray,
    plot_config: DictConfig,
    variable: str,
    ff_averaged_bins: dict[str, np.ndarray],
    ff_combo: dict[str, tuple[float, float]],
    use_pt_met_log: dict[str, bool],
    ff: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))

    if "eta" in variable:
        before_crack = x_range <= 1.37
        after_crack = x_range >= 1.52

        ax.plot(
            x_range[before_crack],
            ff_vals[before_crack],
            color="k",
            label="ML FF",
        )
        ax.fill_between(
            x_range[before_crack],
            ff_vals[before_crack] - ff_errs[before_crack],
            ff_vals[before_crack] + ff_errs[before_crack],
            color="gray",
            alpha=0.5,
            label="ML FF uncertainty",
        )

        ax.plot(
            x_range[after_crack],
            ff_vals[after_crack],
            color="k",
        )
        ax.fill_between(
            x_range[after_crack],
            ff_vals[after_crack] - ff_errs[after_crack],
            ff_vals[after_crack] + ff_errs[after_crack],
            color="gray",
            alpha=0.5,
        )
    else:
        ax.plot(x_range, ff_vals, color="k", label="ML FF")
        ax.fill_between(
            x_range,
            ff_vals - ff_errs,
            ff_vals + ff_errs,
            color="gray",
            alpha=0.5,
            label="ML FF uncertainty",
        )

    if ff is not None:
        binned_x, ff_binned_nom, ff_binned_std = get_binned_ff(variable, ff_averaged_bins, ff_combo, ff)

        x = (binned_x[:-1] + binned_x[1:]) / 2
        xerr = np.diff(binned_x) / 2

        if "eta" in variable:
            crack_idx = ff_binned_nom == 0.0
            x, xerr = x[~crack_idx], xerr[~crack_idx]
            ff_binned_nom, ff_binned_std = ff_binned_nom[~crack_idx], ff_binned_std[~crack_idx]

        ax.errorbar(
            x,
            ff_binned_nom,
            xerr=xerr,
            yerr=ff_binned_std,
            fmt="o",
            ms=4,
            label="Binned FF",
        )
        ax.set_xlim(min(binned_x), max(binned_x))

    xlabel = get_latex_variable_name(variable)

    if variable in ["pt", "met"]:
        xlabel += " [GeV]"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fake Factor")

    title_parts = []
    for k, v in ff_combo.items():
        if k in ["pt", "eta", "met"]:
            if not (k == "eta" and v[0] == 0.0 and v[1] == 2.5):
                title = f"{get_latex_variable_name(k)} in [{v[0]:.2f}, {v[1]:.2f}]"
                if k in ["pt", "met"]:
                    title += " GeV"
                title_parts.append(title)

    ax.set_title(", ".join(title_parts), fontsize=14)

    if variable in ["pt", "met"] and use_pt_met_log.get(variable, False):
        ax.set_xscale("log")

    atlas_marker = plot_config.get("atlas_label", "Internal")

    if atlas_marker is not None:
        atlas_label(ax, loc=0, llabel=atlas_marker)

    ax.legend()

    return fig, ax


def plot_variable_vs_ff(
    dre_result: DensityRatioResult,
    variable: str,
    plot_config: DictConfig,
    ff_averaged_bins: dict[str, np.ndarray],
    custom_ff_averaged_bins: dict[str, np.ndarray] | None = None,
    ff: np.ndarray | None = None,
) -> None:
    figs = []

    mesh = dre_result.mesh
    ml_ff = dre_result.ml_ff

    if custom_ff_averaged_bins is not None:
        ff_averaged_bins_all = custom_ff_averaged_bins.copy()
    else:
        ff_averaged_bins_all = ff_averaged_bins.copy()

    ff_averaged_bins_all["other"] = np.array([0.0, np.inf])

    variable_idx = dre_result.variable_idx
    reversed_variable_idx = {v: k for k, v in variable_idx.items()}

    current_variable_idx = variable_idx[variable]

    others_idx = []
    other_comb_lst = [np.array([])] * 4

    for avg_var, bins in ff_averaged_bins_all.items():
        if avg_var == "abs_eta":
            avg_var = "eta"

        if avg_var == variable:
            continue

        idx = variable_idx[avg_var]
        others_idx.append(idx)
        other_comb_lst[idx] = bins

    others_idx = sorted(others_idx)

    other_comb_lst = [arr for arr in other_comb_lst if arr.size > 0]
    other_centers_idx = [[i for i in range(len(arr) - 1)] for arr in other_comb_lst]
    other_centers_comb = np.array([i for i in itertools.product(*other_centers_idx)])

    integrated_ff = []
    for k in ["pt", "eta", "met"]:
        if k != variable:
            integrated_ff.append(k)

    for center_comb in other_centers_comb:
        bounds_masks, ff_combo = [], {}

        for i, other_comb in enumerate(other_comb_lst):
            edge_l, edge_r = other_comb[center_comb[i]], other_comb[center_comb[i] + 1]
            var_idx = others_idx[i]
            var_take = np.take(mesh, var_idx, axis=1)
            bounds_masks.append((var_take >= edge_l) & (var_take < edge_r))

            ff_var = reversed_variable_idx[var_idx]
            ff_combo[ff_var] = (edge_l, edge_r)

        total_mask = np.column_stack(bounds_masks).all(axis=1)

        if np.all(~total_mask):
            continue

        mesh_masked = mesh[total_mask]
        ml_ff_masked = ml_ff[total_mask]

        var_values = np.take(mesh_masked, current_variable_idx, axis=1).ravel()

        sort_idx = np.argsort(var_values)
        mesh_masked = mesh_masked[sort_idx]
        ml_ff_masked = ml_ff_masked[sort_idx]
        var_values = var_values[sort_idx]

        variable_edges = np.flatnonzero(np.diff(var_values)) + 1
        variable_edges = np.concatenate(([0], variable_edges, [len(var_values)]))

        ff_integrated_vals, ff_integrated_errs, x_range = [], [], []

        for edge in range(len(variable_edges) - 1):
            edge_mask = np.zeros(len(mesh_masked), dtype=bool)
            edge_mask[variable_edges[edge] : variable_edges[edge + 1]] = True

            mesh_integrate = np.take(mesh_masked[edge_mask], current_variable_idx, axis=1)[0]

            ml_ff_integrate = ml_ff_masked[edge_mask]

            ff_mean = np.mean(ml_ff_integrate)
            ff_std = np.std(ml_ff_integrate)

            ff_integrated_vals.append(ff_mean)
            ff_integrated_errs.append(ff_std)
            x_range.append(mesh_integrate)

        fig, _ = plot_integrated_ff(
            np.array(x_range),
            np.array(ff_integrated_vals),
            np.array(ff_integrated_errs),
            plot_config,
            variable,
            ff_averaged_bins,
            ff_combo,
            use_pt_met_log=dre_result.use_pt_met_log,
            ff=ff,
        )
        figs.append(fig)
        plt.close(fig)

    save_path = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "model_ff")
    os.makedirs(save_path, exist_ok=True)

    with PdfPages(os.path.join(save_path, f"ff_avg_{variable}_projection.pdf")) as pdf:
        for fig in tqdm(figs, desc="Saving plots", total=len(figs), leave=False):
            pdf.savefig(fig, bbox_inches="tight")


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    hep.style.use(hep.style.ATLAS)

    plot_config = config.plotting_config.model_ff_plot

    dre_result = get_density_ratio(config)

    ff_file_path = plot_config.get("binned_ff_file_path", None)
    overlay_binned_ff = plot_config.get("compare_methods", True)

    if overlay_binned_ff:
        if ff_file_path is None:
            ff_file_path = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "closure", "binned_ff.p")

        if not os.path.isfile(ff_file_path):
            raise FileNotFoundError(f"Binned ff file not found at {ff_file_path}! Run `fakes_closure` first.")

        ff_dct = load_pickle(ff_file_path)
        ff, ff_averaged_bins = ff_dct["ff"], ff_dct["bins"]
    else:
        ff = None
        ff_averaged_bins = {
            "abs_eta": np.arange(0, 2.75, 0.25),
            "pt": np.linspace(10, 1000, 50),
            "met": np.array([0.0, np.inf]),
        }

    custom_ff_averaged_bins: dict[str, np.ndarray] = {}

    if plot_config.get("ff_averaged_bins", None) is not None:
        for k, v in plot_config.ff_averaged_bins.items():
            custom_ff_averaged_bins[k] = np.array(v)

        unknown_keys = set(custom_ff_averaged_bins.keys()) - set(ff_averaged_bins.keys())
        if len(unknown_keys) > 0:
            raise KeyError(f"Unknown keys in custom ff_averaged_bins: {unknown_keys}")

    variables = ["pt", "eta", "met"]

    for variable in variables:
        plot_variable_vs_ff(
            dre_result,
            variable,
            plot_config,
            ff_averaged_bins,
            custom_ff_averaged_bins=custom_ff_averaged_bins if len(custom_ff_averaged_bins) > 0 else None,
            ff=ff,
        )


if __name__ == "__main__":
    main()
