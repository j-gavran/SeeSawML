import itertools
import logging
import os
from dataclasses import dataclass
from typing import Any

import hydra
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from f9columnar.ml.scalers import FeatureScaler
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig
from tqdm import tqdm

from seesaw.fakes.models.loss import DensityRatio
from seesaw.fakes.training.fakes_trainer import load_ratio_model
from seesaw.fakes.training.tracker_plotting import atlas_label
from seesaw.models.utils import load_reports
from seesaw.utils.features import get_feature
from seesaw.utils.helpers import (
    get_log_binning,
    load_dataset_column,
    load_dataset_column_from_config,
    setup_analysis_dirs,
)
from seesaw.utils.loggers import setup_logger


def get_reports(model_config: DictConfig) -> list[dict[str, Any]]:
    load_checkpoint = model_config.load_checkpoint
    pt_sliced_model = model_config.pt_sliced_model

    if load_checkpoint:
        checkpoint_path = os.path.join(model_config.training_config.model_save_path, load_checkpoint)
        reports = [load_reports(checkpoint_path)]
    elif pt_sliced_model:
        checkpoints = pt_sliced_model.checkpoints
        model_save_path = pt_sliced_model.get("model_save_path", None)
        model_save_paths = pt_sliced_model.get("model_save_paths", None)

        if model_save_path:
            checkpoint_path = os.path.join(model_save_path, checkpoints[0])
            reports = [load_reports(checkpoint_path)]
        elif model_save_paths:
            reports = []
            for checkpoint, save_path in zip(checkpoints, model_save_paths):
                checkpoint_path = os.path.join(save_path, checkpoint)
                reports.append(load_reports(checkpoint_path))
        else:
            raise ValueError("Invalid configuration in pt_sliced_model! Provide model_save_path(s).")

    else:
        raise ValueError("Invalid configuration in ratio model! Provide load_checkpoint or pt_sliced_model.")

    return reports


def get_pt_region(reports: list[dict[str, Any]]) -> np.ndarray:
    pt_cuts = []
    for report in reports:
        pt_cuts.append(report["pt_cut"])

    pt_cuts_arr = np.vstack(pt_cuts)
    pt_cuts_arr = pt_cuts_arr[np.argsort(pt_cuts_arr[:, 0])]

    prev_check = pt_cuts_arr[0, 1]
    for i in range(1, len(pt_cuts)):
        pt_cut = pt_cuts[i]

        now_check = pt_cut[0]

        if prev_check - now_check != 0:
            raise ValueError(f"Must have full coverage of pt region! Error in cuts: {pt_cut}.")

        prev_check = pt_cut[1]

    pt_region = np.array([pt_cuts_arr[0, 0], pt_cuts_arr[-1, 1]])

    return pt_region


def get_binning(
    column_names: list[str], plot_config: DictConfig, pt_cut: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    binning_dct = {}
    use_pt_met_log = {"pt": False, "met": False}

    for column_name in column_names:
        variable_config = plot_config["variables"][column_name]

        x_min, x_max = variable_config.get("x_min", None), variable_config.get("x_max", None)

        use_log = variable_config.get("log", False)
        use_log_scale = variable_config.get("log_scale", False)

        if "pt" in column_name and x_min == "auto":
            if pt_cut is None:
                raise ValueError("pt cut not found!")

            x_min = pt_cut[0]

        if "pt" in column_name and x_max == "auto":
            if pt_cut is None:
                raise ValueError("pt cut not found!")

            x_max = pt_cut[1]

        if "pt" in column_name and use_log_scale:
            use_pt_met_log["pt"] = True

        if "met" in column_name and use_log_scale:
            use_pt_met_log["met"] = True

        nbins = variable_config.get("nbins", None)

        if nbins is not None and use_log:
            binning = get_log_binning(x_min, x_max, nbins).astype(np.float32)

        if nbins is not None and not use_log:
            binning = np.linspace(x_min, x_max, nbins).astype(np.float32)

        bins = variable_config.get("bins", None)

        if bins and nbins:
            raise ValueError("Provide bins as list or nbins as int and not both!")

        if bins is not None:
            binning = np.array(bins)

        binning_dct[column_name] = binning

    return binning_dct, use_pt_met_log


def get_combinations(
    binning_dct: dict[str, np.ndarray], crack_veto: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    base_binning_dct, base_variables_dct = {}, {}

    for k, v in binning_dct.items():
        if "eta" in k:
            if "eta" in base_binning_dct:
                raise ValueError("Found second eta variable!")

            if crack_veto:
                abs_v = np.abs(v)
                v = v[(abs_v < 1.37) | (abs_v > 1.52)]

            base_binning_dct["eta"] = v
            base_variables_dct["eta"] = k

        elif "pt" in k:
            if "pt" in base_binning_dct:
                raise ValueError("Found second pt variable!")

            base_binning_dct["pt"] = v
            base_variables_dct["pt"] = k

        elif "met" in k:
            if "met" in base_binning_dct:
                raise ValueError("Found second met variable!")

            base_binning_dct["met"] = v
            base_variables_dct["met"] = k

        else:
            base_binning_dct["other"] = v
            base_variables_dct["other"] = k

    return base_binning_dct, base_variables_dct


def construct_meshes(
    column_names: list[str], base_binning_dct: dict[str, np.ndarray], base_variables_dct: dict[str, str]
) -> np.ndarray:
    binning_lst = [np.array([])] * len(column_names)
    for binning_name, binning in base_binning_dct.items():
        column_idx = column_names.index(base_variables_dct[binning_name])
        binning_lst[column_idx] = binning

    meshes = list(np.meshgrid(*binning_lst, indexing="xy"))
    mesh = np.column_stack([m.flatten() for m in meshes])

    return mesh


def scale_mesh(
    mesh: np.ndarray,
    categ_scaler: FeatureScaler,
    categ_column_idx: np.ndarray,
    numer_scaler: FeatureScaler,
    numer_column_idx: np.ndarray,
) -> np.ndarray:
    scaled_mesh = []

    if numer_scaler is not None:
        numer_mesh = mesh[:, numer_column_idx]
        scaled_mesh.append(numer_scaler.transform(numer_mesh))

    if categ_scaler is not None:
        categ_mesh = mesh[:, categ_column_idx]
        scaled_mesh.append(categ_scaler.transform(categ_mesh))

    return np.column_stack(scaled_mesh)


def calculate_ff(
    mesh: np.ndarray,
    model: torch.nn.Module,
    categ_scaler: FeatureScaler,
    categ_column_idx: np.ndarray,
    numer_scaler: FeatureScaler,
    numer_column_idx: np.ndarray,
    density_ratio: DensityRatio | None,
) -> np.ndarray:
    scaled_mesh = scale_mesh(mesh, categ_scaler, categ_column_idx, numer_scaler, numer_column_idx)
    scaled_tensor_mesh = torch.from_numpy(scaled_mesh).to(torch.float32).cuda()

    with torch.no_grad():
        if density_ratio is not None:
            ml_ff = density_ratio(model(scaled_tensor_mesh).squeeze())
        else:
            ml_ff = model(scaled_tensor_mesh).squeeze()

    return ml_ff.cpu().numpy()


def get_variable_idx(column_names: list[str]) -> dict[str, int]:
    if len(column_names) > 4:
        raise ValueError("More than 4 variables not supported!")

    idx_dct: dict[str, int] = {}

    for i, column_name in enumerate(column_names):
        if "pt" in column_name:
            idx_dct["pt"] = i
        elif "met" in column_name:
            idx_dct["met"] = i
        elif "eta" in column_name:
            idx_dct["eta"] = i
        else:
            idx_dct["other"] = i

    return idx_dct


def make_title(combination: np.ndarray, combinatoric_variables: list[str], extra: str = "") -> str:
    parts = []
    for c, c_name in zip(combination, combinatoric_variables):
        if "njets" in c_name:
            c = int(c)
        elif "met" in c_name:
            c = f"{c:.2f} GeV"
        elif "pt" in c_name:
            c = f"{c:.2f} GeV"
        else:
            c = f"{c:.2f}"

        parts.append(f"{get_feature(c_name).latex_name}: {c}")

    if extra:
        parts.append(extra)

    return ", ".join(parts)


@dataclass
class DensityRatioResult:
    base_binning_dct: dict[str, np.ndarray]
    base_variables_dct: dict[str, str]
    mesh: np.ndarray
    variable_idx: dict[str, int]
    ml_ff: np.ndarray
    use_pt_met_log: dict[str, bool]
    is_density: bool = True

    def __repr__(self):
        return (
            f"DensityRatioResult(mesh_shape={self.mesh.shape}, "
            f"pt_idx={self.pt_idx}, met_idx={self.met_idx}, eta_idx={self.eta_idx})"
        )

    def __str__(self):
        return self.__repr__()


def get_density_ratio(config: DictConfig) -> DensityRatioResult:
    pt_sliced_model_config = config.model_config.get("pt_sliced_model", None)

    if pt_sliced_model_config is None:
        events_column = load_dataset_column_from_config(config, "events")
    else:
        checkpoints = pt_sliced_model_config.checkpoints

        if pt_sliced_model_config.get("model_save_path", None):
            model_save_paths = [pt_sliced_model_config.model_save_path] * len(checkpoints)
        else:
            model_save_paths = pt_sliced_model_config.model_save_paths

        events_column = load_dataset_column(
            model_save_paths[0],
            run_name=pt_sliced_model_config.checkpoints[0].split("_epoch")[0],
            dataset="events",
        )

    plot_config = config.plotting_config.model_ff_plot

    model, _ = load_ratio_model(config)

    if config.model_config.load_checkpoint:
        model = model.model.eval()  # type: ignore[union-attr]
    else:
        model = model.eval()

    reports = get_reports(config.model_config)

    try:
        pt_region = get_pt_region(reports)
    except Exception:
        logging.info("No pt region found. Setting to None.")
        pt_region = None

    report = reports[0]

    unsorted_column_names = list(set(events_column.used_columns) - set(events_column.extra_columns))
    column_names = [c for c in events_column.used_columns if c in unsorted_column_names]

    variable_idx = get_variable_idx(column_names)

    binning_dct, use_pt_met_log = get_binning(column_names, plot_config, pt_region)

    base_binning_dct, base_variables_dct = get_combinations(
        binning_dct, crack_veto=plot_config.get("crack_veto", False)
    )

    mesh = construct_meshes(column_names, base_binning_dct, base_variables_dct)

    categ_scaler, categ_column_idx = report["categ_scaler"], events_column.offset_categ_columns_idx
    numer_scaler, numer_column_idx = report["scaler"], events_column.offset_numer_columns_idx

    if plot_config.use_density:
        density_ratio = DensityRatio(config.model_config.training_config.loss)
    else:
        density_ratio = None

    ml_ff = calculate_ff(
        mesh=mesh,
        model=model,
        categ_scaler=categ_scaler,
        categ_column_idx=categ_column_idx,
        numer_scaler=numer_scaler,
        numer_column_idx=numer_column_idx,
        density_ratio=density_ratio,
    )

    return DensityRatioResult(
        mesh=mesh,
        base_binning_dct=base_binning_dct,
        base_variables_dct=base_variables_dct,
        variable_idx=variable_idx,
        use_pt_met_log=use_pt_met_log,
        ml_ff=ml_ff,
        is_density=plot_config.use_density,
    )


def plot_density(
    dre_result: DensityRatioResult,
    variables: list[str],
    condition_vars: list[str],
    axis_labels: dict[str, str],
    plot_crack_veto: bool = False,
    atlas_marker: str | None = "Internal",
    other_name: str = "njets",
) -> None:
    base_binning_dct = dre_result.base_binning_dct
    mesh = dre_result.mesh
    ml_ff_mesh = dre_result.ml_ff
    variable_idx = dre_result.variable_idx

    cond_bins = [base_binning_dct[v] for v in condition_vars]
    cond_combinations = np.array(list(itertools.product(*cond_bins)))

    cond_idx = [variable_idx[v] for v in condition_vars]
    cond_cols = [np.take(mesh, idx, axis=1) for idx in cond_idx]
    stacked = np.column_stack(cond_cols if cond_idx[0] < cond_idx[1] else cond_cols[::-1])

    figs = []
    for comb in tqdm(cond_combinations, total=len(cond_combinations), desc="Plotting density", leave=False):
        x_bins, y_bins = (base_binning_dct[v] for v in variables)
        nx, ny = len(x_bins), len(y_bins)

        # apply mask
        comb_mask = np.all(stacked == comb, axis=1)
        selected_mesh = mesh[comb_mask]
        ml_ff_take = ml_ff_mesh[comb_mask]

        # extract coordinates for this condition
        x_vals = np.take(selected_mesh, variable_idx[variables[0]], axis=1)
        y_vals = np.take(selected_mesh, variable_idx[variables[1]], axis=1)

        # build a 2D grid initialized as NaNs
        Z = np.full((nx, ny), np.nan)

        # fill grid by matching each (x,y) to its index
        for xv, yv, val in zip(x_vals, y_vals, ml_ff_take):
            i = np.where(x_bins == xv)[0][0]
            j = np.where(y_bins == yv)[0][0]
            Z[i, j] = val

        # meshgrid for plotting
        X, Y = np.meshgrid(x_bins, y_bins, indexing="ij")

        fig, ax = plt.subplots(figsize=(8, 7))
        heatmap = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
        cbar = plt.colorbar(heatmap, ax=ax)

        cbar.ax.set_title("FF" if dre_result.is_density else "Logits", fontsize=14)

        if plot_crack_veto and "eta" in variables:
            eta_min, eta_max = 1.37, 1.52
            if variables[0] == "eta":
                ax.axvspan(eta_min, eta_max, color="white")
                ax.axvspan(-eta_max, -eta_min, color="white")
            elif variables[1] == "eta":
                ax.axhspan(eta_min, eta_max, color="white")
                ax.axhspan(-eta_max, -eta_min, color="white")

        for var, ax_method in zip(variables, [ax.set_xscale, ax.set_yscale]):
            if var in dre_result.use_pt_met_log and dre_result.use_pt_met_log[var]:
                ax_method("log")

        title = make_title(
            comb, [condition_vars[0], other_name if condition_vars[1] == "other" else condition_vars[1]], extra=""
        )

        ax.set_xlabel(axis_labels[variables[0]])
        ax.set_ylabel(axis_labels[variables[1]])

        ax.set_xlim(min(x_bins), max(x_bins))
        ax.set_ylim(min(y_bins), max(y_bins))

        ax.set_title(title, fontsize=13, loc="right")

        if atlas_marker is not None:
            atlas_label(ax, loc=0, llabel=atlas_marker, fontsize=11)

        ax.set_rasterized(True)
        figs.append(fig)
        plt.close(fig)

    save_path = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "model_ff")
    os.makedirs(save_path, exist_ok=True)

    save_name = f"ff_{'density' if dre_result.is_density else 'logits'}_{variables[0]}_{variables[1]}.pdf"

    with PdfPages(os.path.join(save_path, save_name)) as pdf:
        for fig in tqdm(figs, desc="Saving density plots", total=len(figs), leave=False):
            pdf.savefig(fig, bbox_inches="tight", dpi=300)


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    hep.style.use(hep.style.ATLAS)

    dre_result = get_density_ratio(config)

    plot_config = config.plotting_config.model_ff_plot
    atlas_marker = plot_config.get("atlas_label", "Internal")

    if plot_config.get("pt_eta_density_plot", True):
        plot_density(
            dre_result,
            variables=["pt", "eta"],
            condition_vars=["met", "other"],
            axis_labels={"pt": r"$p_{\mathrm{T}}$ [GeV]", "eta": r"$\eta$", "met": "MET [GeV]"},
            plot_crack_veto=plot_config.get("plot_crack_veto", False),
            atlas_marker=atlas_marker,
            other_name="njets",
        )

    if plot_config.get("pt_met_density_plot", True):
        plot_density(
            dre_result,
            variables=["pt", "met"],
            condition_vars=["eta", "other"],
            axis_labels={"pt": r"$p_{\mathrm{T}}$ [GeV]", "met": "MET [GeV]", "eta": r"$\eta$"},
            plot_crack_veto=plot_config.get("plot_crack_veto", False),
            atlas_marker=atlas_marker,
            other_name="njets",
        )

    if plot_config.get("met_eta_density_plot", True):
        plot_density(
            dre_result,
            variables=["met", "eta"],
            condition_vars=["pt", "other"],
            axis_labels={"met": "MET [GeV]", "eta": r"$\eta$", "pt": r"$p_{\mathrm{T}}$ [GeV]"},
            plot_crack_veto=plot_config.get("plot_crack_veto", False),
            atlas_marker=atlas_marker,
            other_name="njets",
        )


if __name__ == "__main__":
    main()
