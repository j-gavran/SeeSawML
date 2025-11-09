import copy
import logging
import os
from typing import Any

import hist
import hydra
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from f9columnar.histograms import HistogramProcessor
from f9columnar.ml.dataloader_helpers import DatasetColumn
from f9columnar.ml.hdf5_dataloader import WeightedBatchType, events_collate_fn
from f9columnar.ml.lightning_data_module import LightningHdf5DataModule
from f9columnar.utils.helpers import dump_pickle, load_pickle
from lightning.pytorch import seed_everything
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig, open_dict
from plothist import plot_data_model_comparison
from torch.utils.data import DataLoader
from tqdm import tqdm
from uncertainties import unumpy as unp

from seesaw.fakes.models.loss import DensityRatio
from seesaw.fakes.training.fakes_trainer import load_ratio_model
from seesaw.fakes.utils import handle_fakes_dataset, nn_reweight
from seesaw.utils.features import get_feature
from seesaw.utils.helpers import get_log_binning, setup_analysis_dirs
from seesaw.utils.loggers import setup_logger
from seesaw.utils.plots_utils import atlas_label, get_color, save_plot


def get_data_module(dataset_conf: DictConfig, disable_scaling: bool = False) -> LightningHdf5DataModule:
    dataset_kwargs: dict[str, Any]

    dataset_kwargs = {
        "use_data": True,
        "use_mc": True,
        "use_loose": True,
        "use_tight": True,
    }

    feature_scaling_config = dataset_conf.get("feature_scaling", None)

    if not disable_scaling and feature_scaling_config is not None:
        feature_scaling_kwargs = {
            "scaler_type": feature_scaling_config.scaler_type,
            "scaler_path": feature_scaling_config.save_path,
            "scalers_extra_hash": str(dataset_conf.files),
        }
    else:
        logging.info("Feature scaling is disabled.")
        feature_scaling_kwargs = {}

    dataset_kwargs = dataset_kwargs | feature_scaling_kwargs | dict(dataset_conf["dataset_kwargs"])
    dataset_kwargs["setup_func"] = handle_fakes_dataset

    logging.info("Setting up closure data module.")

    dataloader_kwargs = dict(dataset_conf.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null for closure tests.")

    dm = LightningHdf5DataModule(
        "closureDataModule",
        dataset_conf.files,
        dataset_conf.features,
        stage_split_piles=dataset_conf.stage_split_piles,
        shuffle=False,
        collate_fn=events_collate_fn,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    return dm


class BinnedFakesHistogram(HistogramProcessor):
    def __init__(self, events_column: DatasetColumn, binning_dct: DictConfig) -> None:
        super().__init__(name="binnedFakesHistogram", as_data=None, weights_field=None, auto_fill=False)
        self.events_column = events_column

        unsorted_column_names = set(self.events_column.used_columns) - set(self.events_column.extra_columns)
        self.column_names = np.array([str(c) for c in self.events_column.used_columns if c in unsorted_column_names])

        self.hist_names = ["data_tight", "data_loose", "mc_tight", "mc_loose"]
        self.column_names_idx: dict[str, int] = {}

        pt_bins, abs_eta_bins, met_bins = binning_dct.pt_bins, binning_dct.abs_eta_bins, binning_dct.met_bins

        for hist_name in self.hist_names:
            self.make_histNd(
                hist_name,
                {
                    "pt": {"bins": pt_bins},
                    "abs_eta": {"bins": abs_eta_bins},
                    "met": {"bins": met_bins},
                },
            )

    def _get_storage_type(self) -> str:
        return "weight"

    def run(self, batch: WeightedBatchType) -> None:
        X, y, w, y_lt, _ = batch

        numer_idx = self.events_column.offset_numer_columns_idx

        column_names = self.column_names[numer_idx]
        X = X[:, numer_idx]

        X, y, w, y_lt = X.numpy(), y.numpy(), w.numpy(), y_lt.numpy()  # type: ignore

        if len(self.column_names_idx) == 0:
            for i, c in enumerate(column_names):
                if "pt" in c:
                    self.column_names_idx["pt"] = i
                elif "eta" in c:
                    self.column_names_idx["eta"] = i
                elif "met" in c:
                    self.column_names_idx["met"] = i
                else:
                    self.column_names_idx[c] = i

        data_mask, mc_mask = y == 1, y == 0
        tight_mask, loose_mask = y_lt == 1, y_lt == 0

        tight_data_mask = data_mask & tight_mask
        tight_data, tight_data_w = X[tight_data_mask], w[tight_data_mask]

        tight_mc_mask = mc_mask & tight_mask
        tight_mc, tight_mc_w = X[tight_mc_mask], w[tight_mc_mask]

        loose_data_mask = data_mask & loose_mask
        loose_data, loose_data_w = X[loose_data_mask], w[loose_data_mask]

        loose_mc_mask = mc_mask & loose_mask
        loose_mc, loose_mc_w = X[loose_mc_mask], w[loose_mc_mask]

        self.fill_histNd(
            "data_tight",
            [
                tight_data[:, self.column_names_idx["pt"]],
                np.abs(tight_data[:, self.column_names_idx["eta"]]),
                tight_data[:, self.column_names_idx["met"]],
            ],
            weight=tight_data_w,
        )
        self.fill_histNd(
            "data_loose",
            [
                loose_data[:, self.column_names_idx["pt"]],
                np.abs(loose_data[:, self.column_names_idx["eta"]]),
                loose_data[:, self.column_names_idx["met"]],
            ],
            weight=loose_data_w,
        )
        self.fill_histNd(
            "mc_tight",
            [
                tight_mc[:, self.column_names_idx["pt"]],
                np.abs(tight_mc[:, self.column_names_idx["eta"]]),
                tight_mc[:, self.column_names_idx["met"]],
            ],
            weight=tight_mc_w,
        )
        self.fill_histNd(
            "mc_loose",
            [
                loose_mc[:, self.column_names_idx["pt"]],
                np.abs(loose_mc[:, self.column_names_idx["eta"]]),
                loose_mc[:, self.column_names_idx["met"]],
            ],
            weight=loose_mc_w,
        )


def get_binned_closure_hists(
    dl: DataLoader, events_column: DatasetColumn, binning_dct: DictConfig
) -> dict[str, hist.Hist]:
    hist_processor = BinnedFakesHistogram(events_column, binning_dct)

    for batch in tqdm(iter(dl), desc="Processing batches"):
        hist_processor.run(batch)

    return hist_processor.hists


def get_binned_fake_factors(hists_dct: dict[str, hist.Hist]) -> tuple[unp.uarray, dict[str, np.ndarray]]:
    data_tight_h, data_loose_h = hists_dct["data_tight"], hists_dct["data_loose"]
    mc_tight_h, mc_loose_h = hists_dct["mc_tight"], hists_dct["mc_loose"]

    data_tight_val, data_loose_val = data_tight_h.values(), data_loose_h.values()
    mc_tight_val, mc_loose_val = mc_tight_h.values(), mc_loose_h.values()

    data_tight_std, data_loose_std = np.sqrt(data_tight_h.variances()), np.sqrt(data_loose_h.variances())  # type: ignore
    mc_tight_std, mc_loose_std = np.sqrt(mc_tight_h.variances()), np.sqrt(mc_loose_h.variances())  # type: ignore

    data_tight_unp = unp.uarray(data_tight_val, data_tight_std)
    data_loose_unp = unp.uarray(data_loose_val, data_loose_std)

    mc_tight_unp = unp.uarray(mc_tight_val, mc_tight_std)
    mc_loose_unp = unp.uarray(mc_loose_val, mc_loose_std)

    num = data_tight_unp - mc_tight_unp
    denom = data_loose_unp - mc_loose_unp

    ff = np.zeros_like(num)
    mask = unp.nominal_values(denom) != 0.0
    ff[mask] = num[mask] / denom[mask]

    bins = {
        "pt": data_tight_h.axes["pt"].edges,
        "abs_eta": data_tight_h.axes["abs_eta"].edges,
        "met": data_tight_h.axes["met"].edges,
    }

    return ff, bins


class BinnedFakeFactorIndexer:
    def __init__(self, ff: unp.uarray, bins: dict[str, np.ndarray]):
        self.ff = ff
        self.bins = bins

    def get_bin_indices(
        self, pt: np.ndarray, abs_eta: np.ndarray, met: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pt_idx = np.clip(np.digitize(pt, self.bins["pt"]) - 1, 0, len(self.bins["pt"]) - 2)
        abs_eta_idx = np.clip(np.digitize(abs_eta, self.bins["abs_eta"]) - 1, 0, len(self.bins["abs_eta"]) - 2)
        met_idx = np.clip(np.digitize(met, self.bins["met"]) - 1, 0, len(self.bins["met"]) - 2)
        return pt_idx, abs_eta_idx, met_idx

    def get(self, pt: np.ndarray, abs_eta: np.ndarray, met: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pt_idx, abs_eta_idx, met_idx = self.get_bin_indices(pt, abs_eta, met)

        ff_value = self.ff[pt_idx, abs_eta_idx, met_idx]
        ff_nom, ff_std = unp.nominal_values(ff_value), unp.std_devs(ff_value)

        return ff_nom, ff_std


def _get_binned_closure_hists(
    batch: WeightedBatchType,
    binned_ff_idx: BinnedFakeFactorIndexer,
    events_column: DatasetColumn,
    names_idx: dict[str, int],
    closure_hists: dict[str, dict[str, hist.Hist]],
) -> dict[str, dict[str, hist.Hist]]:
    X_t, y_t, w_t, y_lt_t, reports = batch
    X, y, w, y_lt = X_t.numpy(), y_t.numpy(), w_t.numpy(), y_lt_t.numpy()

    numer_idx = events_column.offset_numer_columns_idx

    unsorted_column_names = set(events_column.used_columns) - set(events_column.extra_columns)
    column_names = np.array([str(c) for c in events_column.used_columns if c in unsorted_column_names])[numer_idx]

    scaler = reports["scaler"]

    # find the indices of pt, eta, met columns
    pt_idx, met_idx, eta_idx = names_idx["pt"], names_idx["met"], names_idx["eta"]

    # just consider the numerical columns
    X = X[:, numer_idx]

    # need to scale the data back to the original values where ff was calculated
    if scaler is not None:
        X = scaler.inverse_transform(X)

    data_mask, mc_mask = y == 1, y == 0
    tight_mask, loose_mask = y_lt == 1, y_lt == 0

    tight_data_mask = data_mask & tight_mask
    tight_data, tight_data_w = X[tight_data_mask], w[tight_data_mask]

    tight_mc_mask = mc_mask & tight_mask
    tight_mc, tight_mc_w = X[tight_mc_mask], w[tight_mc_mask]

    loose_data_mask = data_mask & loose_mask
    loose_data, loose_data_w = X[loose_data_mask], w[loose_data_mask]

    loose_mc_mask = mc_mask & loose_mask
    loose_mc, loose_mc_w = X[loose_mc_mask], w[loose_mc_mask]

    binned_loose_data_ff, _ = binned_ff_idx.get(
        loose_data[:, pt_idx],
        np.abs(loose_data[:, eta_idx]),
        loose_data[:, met_idx],
    )
    binned_loose_mc_ff, _ = binned_ff_idx.get(
        loose_mc[:, pt_idx],
        np.abs(loose_mc[:, eta_idx]),
        loose_mc[:, met_idx],
    )

    for i, column_name in enumerate(column_names):
        if column_name in closure_hists:
            h_dct = closure_hists[column_name]

            tight_data_i, tight_mc_i = tight_data[:, i], tight_mc[:, i]
            loose_data_i, loose_mc_i = loose_data[:, i], loose_mc[:, i]

            h_dct["binned_tight_data"].fill(tight_data_i, weight=tight_data_w)
            h_dct["binned_tight_mc"].fill(tight_mc_i, weight=tight_mc_w)

            h_dct["binned_loose_data"].fill(loose_data_i, weight=loose_data_w * binned_loose_data_ff)
            h_dct["binned_loose_mc"].fill(loose_mc_i, weight=-loose_mc_w * binned_loose_mc_ff)

    return closure_hists


def _get_nn_closure_hists(
    batch: WeightedBatchType,
    model: torch.nn.Module,
    events_column: DatasetColumn,
    density_ratio: DensityRatio,
    closure_hists: dict[str, dict[str, hist.Hist]],
) -> dict[str, dict[str, hist.Hist]]:
    X_t, y_t, w_t, y_lt_t, reports = batch
    X, y, w, y_lt = X_t, y_t.numpy(), w_t.numpy(), y_lt_t.numpy()

    numer_idx, categ_idx = events_column.offset_numer_columns_idx, events_column.offset_categ_columns_idx
    column_names = sorted(set(events_column.used_columns) - set(events_column.extra_columns))
    scaler = reports["scaler"]

    data_mask, mc_mask = y == 1, y == 0
    tight_mask, loose_mask = y_lt == 1, y_lt == 0

    tight_data_mask = data_mask & tight_mask
    tight_data, tight_data_w = X[tight_data_mask].numpy(), w[tight_data_mask]

    tight_mc_mask = mc_mask & tight_mask
    tight_mc, tight_mc_w = X[tight_mc_mask].numpy(), w[tight_mc_mask]

    loose_data_mask = data_mask & loose_mask
    # assume model on gpu and put tensor on cuda
    loose_data, loose_data_w = X[loose_data_mask].cuda(), w[loose_data_mask]

    loose_mc_mask = mc_mask & loose_mask
    # assume model on gpu and put tensor on cuda
    loose_mc, loose_mc_w = X[loose_mc_mask].cuda(), w[loose_mc_mask]

    loose_data_nn_w = nn_reweight(model, loose_data, density_ratio).cpu().numpy().flatten()  # type: ignore
    loose_mc_nn_w = nn_reweight(model, loose_mc, density_ratio).cpu().numpy().flatten()  # type: ignore

    # fill nn weights histograms
    closure_hists["nn_weight"]["loose_data_weight"].fill(loose_data_nn_w)
    closure_hists["nn_weight"]["loose_mc_weight"].fill(loose_mc_nn_w)

    loose_data, loose_mc = loose_data.cpu().numpy(), loose_mc.cpu().numpy()  # type: ignore

    # handle scaling
    if scaler is not None:
        # if categ_idx, we need to separate the categorical and numerical columns
        if len(categ_idx) != 0:
            loose_categ_data, loose_categ_mc = loose_data[:, categ_idx], loose_mc[:, categ_idx]
            tight_categ_data, tight_categ_mc = tight_data[:, categ_idx], tight_mc[:, categ_idx]

            loose_numer_data, loose_numer_mc = loose_data[:, numer_idx], loose_mc[:, numer_idx]
            tight_numer_data, tight_numer_mc = tight_data[:, numer_idx], tight_mc[:, numer_idx]

            # scale the numerical columns back to the original values
            # TODO: also scale back categorical columns
            loose_numer_data = scaler.inverse_transform(loose_numer_data)
            loose_numer_mc = scaler.inverse_transform(loose_numer_mc)

            tight_numer_data = scaler.inverse_transform(tight_numer_data)
            tight_numer_mc = scaler.inverse_transform(tight_numer_mc)

            # concatenate the categorical and numerical columns
            loose_data = np.concatenate([loose_categ_data, loose_numer_data], axis=1)  # type: ignore
            loose_mc = np.concatenate([loose_categ_mc, loose_numer_mc], axis=1)  # type: ignore

            tight_data = np.concatenate([tight_categ_data, tight_numer_data], axis=1)
            tight_mc = np.concatenate([tight_categ_mc, tight_numer_mc], axis=1)

            categ_column_names = list(np.array(column_names)[categ_idx])
            numer_column_names = list(np.array(column_names)[numer_idx])

            # correct the column names order because we concatenated the columns
            column_names = categ_column_names + numer_column_names
        else:
            loose_data, loose_mc = scaler.inverse_transform(loose_data), scaler.inverse_transform(loose_mc)
            tight_data, tight_mc = scaler.inverse_transform(tight_data), scaler.inverse_transform(tight_mc)

    for i, column_name in enumerate(column_names):
        if column_name in closure_hists:
            h_dct = closure_hists[column_name]

            loose_data_i, loose_mc_i = loose_data[:, i], loose_mc[:, i]
            tight_data_i, tight_mc_i = tight_data[:, i], tight_mc[:, i]

            h_dct["tight_data"].fill(tight_data_i, weight=tight_data_w)
            h_dct["tight_mc"].fill(tight_mc_i, weight=tight_mc_w)

            h_dct["loose_data"].fill(loose_data_i, weight=loose_data_w * loose_data_nn_w)
            h_dct["loose_mc"].fill(loose_mc_i, weight=-loose_mc_w * loose_mc_nn_w)

    return closure_hists


def get_closure_hists(
    dl: DataLoader,
    events_column: DatasetColumn,
    plotting_conf: DictConfig,
    binned_ff_idx: BinnedFakeFactorIndexer | None = None,
    model: torch.nn.Module | None = None,
    density_ratio: DensityRatio | None = None,
) -> dict[str, dict[str, hist.Hist]]:
    # dict to store histograms, {plotting_var: {...t/l/data/mc/binned hists dicts...}, ...}
    closure_hists: dict[str, dict[str, hist.Hist]] = {}

    # setup histograms for closure variables
    for plotting_var, plotting_var_conf in plotting_conf.variables.items():
        if plotting_var == "nn_weight":
            continue

        logx = plotting_var_conf.get("logx", False)

        if "bins" in plotting_var_conf:
            bins = plotting_var_conf.bins
        else:
            if logx:
                bins = get_log_binning(plotting_var_conf.x_min, plotting_var_conf.x_max, plotting_var_conf.nbins)
            else:
                bins = np.linspace(plotting_var_conf.x_min, plotting_var_conf.x_max, plotting_var_conf.nbins)

        closure_hists[plotting_var] = {}

        binned_tight_data_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())
        binned_loose_data_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())

        closure_hists[plotting_var]["binned_tight_data"] = binned_tight_data_h
        closure_hists[plotting_var]["binned_loose_data"] = binned_loose_data_h

        binned_tight_mc_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())
        binned_loose_mc_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())

        closure_hists[plotting_var]["binned_tight_mc"] = binned_tight_mc_h
        closure_hists[plotting_var]["binned_loose_mc"] = binned_loose_mc_h

        tight_data_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())
        tight_mc_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())

        closure_hists[plotting_var]["tight_data"] = tight_data_h
        closure_hists[plotting_var]["tight_mc"] = tight_mc_h

        loose_data_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())
        loose_mc_h = hist.Hist(hist.axis.Variable(bins, name="closure"), storage=hist.storage.Weight())

        closure_hists[plotting_var]["loose_data"] = loose_data_h
        closure_hists[plotting_var]["loose_mc"] = loose_mc_h

    # setup weights histogram for nn weights
    nn_weight_conf = plotting_conf.variables["nn_weight"]
    if "bins" in nn_weight_conf:
        bins = nn_weight_conf.bins
    else:
        bins = np.linspace(nn_weight_conf.x_min, nn_weight_conf.x_max, nn_weight_conf.nbins)

    closure_hists["nn_weight"] = {}
    closure_hists["nn_weight"]["loose_data_weight"] = hist.Hist(hist.axis.Variable(bins, name="closure"))
    closure_hists["nn_weight"]["loose_mc_weight"] = hist.Hist(hist.axis.Variable(bins, name="closure"))

    # only need numerical column names for binned closure
    numer_column_names_idx: dict[str, int] = {}

    unsorted_column_names = set(events_column.used_columns) - set(events_column.extra_columns)
    column_names = np.array([str(c) for c in events_column.used_columns if c in unsorted_column_names])

    numer_idx = events_column.offset_numer_columns_idx
    numer_column_names = column_names[numer_idx]

    for batch in tqdm(iter(dl), desc="Processing closure"):
        if len(numer_column_names_idx) == 0:
            for i, c in enumerate(numer_column_names):
                if "pt" in c:
                    numer_column_names_idx["pt"] = i
                elif "eta" in c:
                    numer_column_names_idx["eta"] = i
                elif "met" in c:
                    numer_column_names_idx["met"] = i

        if binned_ff_idx is not None:
            closure_hists = _get_binned_closure_hists(
                batch,
                binned_ff_idx,
                events_column,
                numer_column_names_idx,
                closure_hists,
            )
        if model is not None and density_ratio is not None:
            closure_hists = _get_nn_closure_hists(
                batch,
                model,
                events_column,
                density_ratio,
                closure_hists,
            )

    return closure_hists


def plot_binned_ff(
    save_dir: str, pt_cut: dict[str, float] | None = None, atlas_marker: str | None = "Internal"
) -> None:
    ff_dct = load_pickle(f"{save_dir}/binned_ff.p")
    binned_ff = ff_dct["ff"]
    hists = load_pickle(f"{save_dir}/closure_hists.p")

    eta_bins = ff_dct["bins"]["abs_eta"]
    met_bins = ff_dct["bins"]["met"]

    w = hists["data_tight"].axes[0].widths
    c = hists["data_tight"].axes[0].centers

    eta_indices = [i for i in range(len(eta_bins) - 1)]
    met_indices = [i for i in range(len(met_bins) - 1)]

    fig, axs = plt.subplots(1, len(met_indices), figsize=(6 * len(met_indices), 5.5))
    if type(axs) is not np.ndarray:
        axs = np.array([axs])

    for j, eta_idx in enumerate(eta_indices):
        for i, met_idx in enumerate(met_indices):
            ff_pt = binned_ff[:, eta_idx, met_idx]
            ff_pt_nom, ff_pt_std = unp.nominal_values(ff_pt), unp.std_devs(ff_pt)

            eta_l = rf"{eta_bins[eta_idx]} $<|\eta|<$ {eta_bins[eta_idx + 1]}"
            if i < len(met_indices) - 1:
                met_l = rf", {met_bins[met_idx]} $<E^{{\mathrm{{miss}}}}_{{\mathrm{{T}}}}<$ {met_bins[met_idx + 1]} GeV"
            else:
                met_l = rf", $E^{{\mathrm{{miss}}}}_{{\mathrm{{T}}}}>$ {met_bins[met_idx]} GeV"

            label = eta_l + met_l

            if np.all(ff_pt_nom == 0.0):
                continue

            axs[i].errorbar(c, ff_pt_nom, xerr=w / 2, yerr=ff_pt_std, fmt="o", ms=4, lw=1.5, color=f"C{j}", label=label)

            logging.info(f"Fake Factor details for MET bin {i}, eta bin {j}:")
            for pt_nom, pt_std in zip(ff_pt_nom, ff_pt_std):
                logging.info(f"  {pt_nom:.3f} ± {pt_std:.3f}")

    for ax in axs:
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlabel(r"$p_{\mathrm{T}}$ [GeV]", fontsize=16)
        ax.set_ylabel("Fake Factor", fontsize=15)
        ax.set_xscale("log")

        ax.set_xlim(c[0] - w[0] / 2, c[-1] + w[-1] / 2)

        if pt_cut is not None:
            ax.set_xlim(pt_cut["min"], pt_cut["max"])

    if atlas_marker is not None:
        atlas_label(axs[0], loc=0, llabel=atlas_marker)

    save_plot(fig, f"{save_dir}/binned_ff.pdf")


def plot_nn_weights(
    closure_hists: dict[str, dict[str, hist.Hist]],
    plotting_conf: DictConfig,
    save_dir: str,
    atlas_marker: str | None = "Internal",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 5))

    data_weight_h = closure_hists["nn_weight"]["loose_data_weight"]
    mc_weight_h = closure_hists["nn_weight"]["loose_mc_weight"]

    hep.histplot(
        [data_weight_h, mc_weight_h],
        ax=ax,
        histtype="step",
        label=["Loose data", "Loose MC"],
        color=[get_color("Red").rgb, get_color("Blue").rgb],
        density=True,
        lw=1.2,
    )

    ax.set_xlabel("NN weight", fontsize=14)
    ax.set_ylabel("Events", fontsize=14)

    ax.legend()

    if plotting_conf.variables["nn_weight"].get("logy", False):
        ax.set_yscale("log")

    ax.set_ylim(bottom=1e-3)

    if atlas_marker is not None:
        atlas_label(ax, loc=0, llabel=atlas_marker)

    save_plot(fig, f"{save_dir}/nn_weights.pdf")


def plot_closure_hists(
    closure_hists: dict[str, dict[str, hist.Hist]],
    plotting_conf: DictConfig,
    save_dir: str,
    is_ff_binned: bool = False,
    atlas_marker: str | None = "Internal",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    color = [get_color("Blue").rgb, get_color("Pink").rgb]

    figs = []
    for variable_name, hists in closure_hists.items():
        if variable_name not in plotting_conf.variables:
            logging.warning(f"Variable {variable_name} not found in the plotting config. Skipping.")
            continue

        if variable_name == "nn_weight":
            continue

        variable_plotting_conf = plotting_conf.variables[variable_name]

        if is_ff_binned:
            tight_data_h, tight_mc_h = hists["binned_tight_data"], hists["binned_tight_mc"]
            loose_data_h, loose_mc_h = hists["binned_loose_data"], hists["binned_loose_mc"]
        else:
            tight_data_h, tight_mc_h = hists["tight_data"], hists["tight_mc"]
            loose_data_h, loose_mc_h = hists["loose_data"], hists["loose_mc"]

        fake_counts = loose_data_h + loose_mc_h

        fig, ax_main, ax_ratio = plot_data_model_comparison(
            data_hist=tight_data_h,
            stacked_components=[tight_mc_h, fake_counts],
            stacked_labels=["MC", "Fakes"],
            stacked_colors=color,
            xlabel=str(get_feature(variable_name)),
            ylabel="Events / Bin",
            comparison="split_ratio",
            model_uncertainty=True,
            data_uncertainty_type="symmetrical",
        )
        fig.set_size_inches(7, 7)

        ax_main.set_xlabel("")
        ax_main.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        ax_main.set_ylim(0, np.max(tight_data_h.values()) * 1.2)

        ax_main.legend()

        if variable_plotting_conf.get("logx", False):
            ax_main.set_xscale("log")
            ax_ratio.set_xscale("log")

        if "x_min" in variable_plotting_conf and "x_max" in variable_plotting_conf:
            x_min, x_max = variable_plotting_conf["x_min"], variable_plotting_conf["x_max"]
            ax_main.set_xlim(x_min, x_max)
            ax_ratio.set_xlim(x_min, x_max)

        ax_ratio.set_ylim(0.8, 1.2)

        if is_ff_binned:
            ax_main.set_title("Binned fake factors", fontsize=14, loc="right")
        else:
            ax_main.set_title("ML fake factors", fontsize=14, loc="right")

        if atlas_marker is not None:
            atlas_label(ax_main, fontsize=12, loc=1, llabel=atlas_marker)

        figs.append(copy.deepcopy(fig))

        ax_main.set_ylim(1, np.max(tight_data_h.values()) * 10)
        ax_main.set_yscale("log")

        figs.append(fig)

    if is_ff_binned:
        save_name = f"{save_dir}/binned_ff_closure.pdf"
    else:
        save_name = f"{save_dir}/ml_ff_closure.pdf"

    with PdfPages(save_name) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    hep.style.use(hep.style.ATLAS)
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    save_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "closure")

    with open_dict(config):
        config.dataset_config.dataset_kwargs.drop_last = False

    dataset_config, model_config = config.dataset_config, config.model_config
    plotting_config = config.plotting_config["closure_plot"]

    atlas_marker = plotting_config.get("atlas_label", "Internal")

    closure_type = plotting_config.get("closure_type", "all")
    if closure_type not in ["binned", "ml", "all"]:
        raise ValueError(f"Invalid closure type: {closure_type}. Must be 'binned', 'ml', or 'all'.")

    logging.info(f"Closure test mode: [green]{closure_type}[/green].")

    os.makedirs(save_dir, exist_ok=True)

    if (closure_type == "all" or closure_type == "ml") and model_config.name != "ratioModel":
        raise ValueError("Model should be a ratioModel!")

    n_val_piles, n_test_piles = dataset_config.stage_split_piles["val"], dataset_config.stage_split_piles["test"]
    if n_test_piles == 0:
        if n_val_piles == 0:
            raise ValueError("No validation or test piles found in the dataset configuration!")
        logging.warning("No test piles found, using validation piles for testing.")
        stage = "val"
    else:
        stage = "test"

    if closure_type == "binned" or closure_type == "all":
        logging.info("Calculating binned fake factors.")

        seed_everything(config.experiment_config.seed, workers=True)
        dm = get_data_module(dataset_config, disable_scaling=True)
        dm.setup(stage=stage)

        if stage == "test":
            dl = dm.test_dataloader()
        else:
            dl = dm.val_dataloader()

        events_column: DatasetColumn = dm.selection["events"]

        hists_dct = get_binned_closure_hists(dl, events_column, plotting_config.ff_standard_binning)

        logging.info("Saving hists dict.")
        dump_pickle(f"{save_dir}/closure_hists.p", hists_dct)

        binned_ff, ff_bins = get_binned_fake_factors(hists_dct)

        logging.info("Saving binned fake factors.")
        dump_pickle(f"{save_dir}/binned_ff.p", {"ff": binned_ff, "bins": ff_bins})

        plot_binned_ff(save_dir, config.dataset_config.dataset_kwargs.get("pt_cut", None), atlas_marker)

        logging.info("Calculating closure histograms.")

        binned_ff_idx = BinnedFakeFactorIndexer(binned_ff, ff_bins)
    else:
        binned_ff_idx = None

    if closure_type == "ml" or closure_type == "all":
        logging.info("Calulating ML fake factors.")

        seed_everything(config.experiment_config.seed, workers=True)
        dm = get_data_module(dataset_config, disable_scaling=False)
        dm.setup(stage=stage)

        if stage == "test":
            dl = dm.test_dataloader()
        else:
            dl = dm.val_dataloader()

        events_column: DatasetColumn = dm.selection["events"]  # type: ignore[no-redef]

        model, _ = load_ratio_model(config)
        model = model.cuda()

        density_ratio = DensityRatio(model_config.training_config.loss)
    else:
        model, density_ratio = None, None

    closure_hists = get_closure_hists(
        dl,
        events_column,
        plotting_config,
        binned_ff_idx=binned_ff_idx,
        model=model,
        density_ratio=density_ratio,
    )

    if closure_type == "binned" or closure_type == "all":
        logging.info("Plotting binned fake factor closure histograms.")
        plot_closure_hists(closure_hists, plotting_config, save_dir, is_ff_binned=True, atlas_marker=atlas_marker)

    if closure_type == "ml" or closure_type == "all":
        logging.info("Plotting ML fake factor closure histograms.")
        plot_nn_weights(closure_hists, plotting_config, save_dir, atlas_marker=atlas_marker)
        plot_closure_hists(closure_hists, plotting_config, save_dir, is_ff_binned=False, atlas_marker=atlas_marker)


if __name__ == "__main__":
    main()
