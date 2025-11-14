from __future__ import annotations

import os

import hist
import mplhep as hep
import numpy as np
from f9columnar.ml.dataloader_helpers import DatasetColumn, column_selection_from_dict
from f9columnar.ml.hdf5_dataloader import WeightedBatchType
from f9columnar.utils.helpers import load_json
from omegaconf import DictConfig

from seesaw.fakes.models.loss import DensityRatio
from seesaw.fakes.training.tracker_plotting import (
    plot_num_den_hists,
    plot_num_den_weights,
    plot_ratio_distributions,
)
from seesaw.fakes.utils import get_num_den_weights
from seesaw.models.tracker import Tracker
from seesaw.utils.helpers import get_log_binning, to_cpu_numpy


class NumDenTracker(Tracker):
    def __init__(
        self,
        experiment_conf: DictConfig,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        plotting_conf: DictConfig,
        tracker_path: str,
        num_den: str,
    ) -> None:
        super().__init__(experiment_conf, tracker_path)
        if num_den not in ["num", "den"]:
            raise ValueError("num_den should be either 'num' or 'den'")

        self.dataset_conf = dataset_conf
        self.model_conf = model_conf
        self.plotting_conf = plotting_conf
        self.num_den = num_den
        self.tracker_epoch = -1

        self.density_ratio = DensityRatio(name=self.model_conf["training_config"]["loss"])

        self.events_column: DatasetColumn | None = None

        self.hists: dict[str, dict[str, hist.Hist]]
        self.hists_info: dict[str, dict[str, dict[str, bool | int]]]
        self.weights_hists: dict[str, hist.Hist]

    def _load_events_column(self) -> DatasetColumn:
        selection_path = os.path.join(
            self.model_conf.training_config.model_save_path, f"{self.experiment_conf.run_name}_selection.json"
        )
        selection_dct = load_json(selection_path)

        selection = column_selection_from_dict(selection_dct)

        return selection["events"]

    def _get_all_weights_bins(self, name: str) -> np.ndarray:
        plot_conf = self.plotting_conf["subtraction_plot"]["weights"][name]
        return np.linspace(plot_conf[0], plot_conf[1], plot_conf[2])

    def _all_weights(self) -> None:
        weights_hs = [
            "data_prescales",
            "mc_weights",
            "data_out",
            "mc_out",
            "data_density",
            "mc_density",
            "data_sub",
            "mc_sub",
            "data_reweighted",
            "mc_reweighted",
        ]

        hs = {}
        for h_name in weights_hs:
            hs[h_name] = self._get_all_weights_bins(h_name)

        for h_name, bins in hs.items():
            h = hist.Hist(hist.axis.Variable(bins, name=h_name))
            self.weights_hists[h_name] = h

    def _data_mc_reweighted(self, plot_column: str, column_names: list[str]) -> None:
        plot_conf = self.plotting_conf["subtraction_plot"]["variables"][plot_column]

        x_min, x_max, nbins = plot_conf["x_min"], plot_conf["x_max"], plot_conf["nbins"]

        if plot_conf["logx"] is True:
            bins = get_log_binning(x_min, x_max, nbins)
        else:
            bins = np.linspace(x_min, x_max, nbins)

        for data_mc in [True, False]:
            for is_reweighted in [False, True]:
                if data_mc:
                    k = "data"
                else:
                    k = "mc"

                if is_reweighted:
                    k += "_reweighted"

                self.hists_info[plot_column][k] = {"is_data": False, "is_reweighted": False, "idx": 0}

                self.hists_info[plot_column][k]["is_data"] = True if data_mc else False
                self.hists_info[plot_column][k]["is_reweighted"] = True if is_reweighted else False
                self.hists_info[plot_column][k]["idx"] = column_names.index(plot_column)

                h = hist.Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
                self.hists[plot_column][k] = h

                if plot_conf["logx"] is True:
                    setattr(h, "logx", True)
                else:
                    setattr(h, "logx", False)

    def on_new_epoch(self) -> None:
        if self.events_column is None:
            self.events_column = self._load_events_column()

        self.weights_hists = {}
        self._all_weights()

        self.hists, self.hists_info = {}, {}

        column_names = sorted(
            set(self.events_column.used_columns)
            - set(self.events_column.categ_columns)
            - set(self.events_column.extra_columns)
        )
        plot_columns = list(self.plotting_conf["subtraction_plot"]["variables"].keys())

        for plot_column in plot_columns:
            plot_column = str(plot_column)

            if plot_column not in column_names:
                raise RuntimeError(f"Variable {plot_column} not found in the plotting configuration!")

            self.hists[plot_column], self.hists_info[plot_column] = {}, {}
            self._data_mc_reweighted(plot_column, column_names)

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {"hists": f"{self.base_dir}/hists/"}

    def get_predictions(self, batch: WeightedBatchType) -> None:
        X, y, w, _, reports = batch

        data_label, mc_label = 1, 0

        data_mask, mc_mask = y == data_label, y == mc_label

        X_data, X_mc = X[data_mask], X[mc_mask]
        w_data, w_mc = w[data_mask], w[mc_mask]

        w_nn_data, density_data, f_data = get_num_den_weights(
            self.module,
            X_data,
            is_data=True,
            density_ratio=self.density_ratio,
            return_intermediate=True,
        )
        w_nn_mc, density_mc, f_mc = get_num_den_weights(
            self.module,
            X_mc,
            is_data=False,
            density_ratio=self.density_ratio,
            return_intermediate=True,
        )

        numer_idx = self.events_column.offset_numer_columns_idx  # type: ignore
        X_data, X_mc = X_data[:, numer_idx], X_mc[:, numer_idx]

        w_nn_data, w_nn_mc = w_nn_data.flatten(), w_nn_mc.flatten()
        density_mc, density_data = density_mc.flatten(), density_data.flatten()
        f_data, f_mc = f_data.flatten(), f_mc.flatten()

        X_data, X_mc, w_data, w_mc = to_cpu_numpy(X_data, X_mc, w_data, w_mc)  # type: ignore
        w_nn_data, w_nn_mc = to_cpu_numpy(w_nn_data, w_nn_mc)  # type: ignore
        density_data, density_mc = to_cpu_numpy(density_data, density_mc)  # type: ignore
        f_data, f_mc = to_cpu_numpy(f_data, f_mc)  # type: ignore

        w_nn_data, w_nn_mc = np.nan_to_num(w_nn_data), np.nan_to_num(w_nn_mc)  # type: ignore

        feature_scaler = reports["numer_scaler"]
        if feature_scaler is not None:
            if X_data.shape[0] >= 1:
                X_data = feature_scaler.inverse_transform(X_data)

            if X_mc.shape[0] >= 1:
                X_mc = feature_scaler.inverse_transform(X_mc)

        self.weights_hists["data_prescales"].fill(w_data)
        self.weights_hists["mc_weights"].fill(w_mc)
        self.weights_hists["data_out"].fill(f_data)
        self.weights_hists["mc_out"].fill(f_mc)
        self.weights_hists["data_density"].fill(density_data)
        self.weights_hists["mc_density"].fill(density_mc)
        self.weights_hists["data_sub"].fill(w_nn_data)
        self.weights_hists["mc_sub"].fill(w_nn_mc)
        self.weights_hists["data_reweighted"].fill(w_data * w_nn_data)
        self.weights_hists["mc_reweighted"].fill(w_mc * w_nn_mc)

        for plot_column, plot_column_dct in self.hists.items():
            for k, h in plot_column_dct.items():
                h_info = self.hists_info[plot_column][k]

                idx = h_info["idx"]

                if h_info["is_reweighted"]:
                    if h_info["is_data"]:
                        h.fill(X_data[:, idx], weight=w_data * w_nn_data)
                    else:
                        h.fill(X_mc[:, idx], weight=w_mc * w_nn_mc)
                else:
                    if h_info["is_data"]:
                        h.fill(X_data[:, idx], weight=w_data)
                    else:
                        h.fill(X_mc[:, idx], weight=w_mc)

    def get_binned_chi2(self) -> tuple[float, float]:
        chi2_data_lst, chi2_mc_list = [], []

        for k in self.hists.keys():
            h_data, h_mc = self.hists[k]["data"], self.hists[k]["mc"]
            h_data_reweighted, h_mc_reweighted = self.hists[k]["data_reweighted"], self.hists[k]["mc_reweighted"]

            reference_sub = np.abs(h_data.values() - h_mc.values()) + 1.0e-8
            nn_data_sub, nn_mc_sub = h_data_reweighted.values(), h_mc_reweighted.values()

            chi2_data = np.sum((nn_data_sub - reference_sub) ** 2 / reference_sub)
            chi2_mc = np.sum((nn_mc_sub - reference_sub) ** 2 / reference_sub)

            chi2_data_lst.append(chi2_data)
            chi2_mc_list.append(chi2_mc)

        return np.mean(chi2_data_lst), np.mean(chi2_mc_list)

    def compute(self, batch: WeightedBatchType, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if not self.validate_compute():
            return False

        if self.current_epoch != self.tracker_epoch:
            self.on_new_epoch()
            self.tracker_epoch = self.current_epoch

        self.get_predictions(batch)

        return True

    def plot(self, stage: str) -> bool:
        if not self.validate_plot():
            return False

        hep.style.use(hep.style.ATLAS)

        plot_num_den_weights(
            self.weights_hists,
            save_dir=self.plotting_dirs["hists"],
            save_prefix=f"{self.num_den}_{stage}_{self.current_epoch}_",
            atlas_marker=self.plotting_conf.subtraction_plot.get("atlas_label", "Internal"),
        )
        plot_num_den_hists(
            self.hists,
            self.num_den,
            save_dir=self.plotting_dirs["hists"],
            save_prefix=f"{self.num_den}_{stage}_{self.current_epoch}_",
            atlas_marker=self.plotting_conf.subtraction_plot.get("atlas_label", "Internal"),
        )

        self.log_artifacts()

        return True

    def reset(self):
        return super().reset()


class RatioTracker(Tracker):
    def __init__(
        self,
        experiment_conf: DictConfig,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        plotting_conf: DictConfig,
        tracker_path: str,
    ) -> None:
        super().__init__(experiment_conf, tracker_path)

        self.dataset_conf = dataset_conf
        self.model_conf = model_conf
        self.plotting_conf = plotting_conf
        self.tracker_epoch = -1

        self.density_ratio = DensityRatio(name=self.model_conf["training_config"]["loss"])

        self.dists: dict[str, hist.Hist] = {}

    def on_new_epoch(self) -> None:
        self.dists["logits"] = hist.Hist(hist.axis.Variable(np.linspace(-10.0, 10.0, 100), name="logits"))
        self.dists["density"] = hist.Hist(hist.axis.Variable(np.linspace(-1.0, 20.0, 100), name="density"))

    def get_predictions(self, batch: WeightedBatchType) -> None:
        X = batch[0]

        y_hat = self.module(X).flatten()
        y_ratio = self.density_ratio(y_hat)

        self.dists["density"].fill(y_ratio.cpu().numpy())
        self.dists["logits"].fill(y_hat.cpu().numpy())

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {"dist": f"{self.base_dir}/dist/"}

    def compute(self, batch: WeightedBatchType, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if not self.validate_compute():
            return False

        if self.current_epoch != self.tracker_epoch:
            self.on_new_epoch()
            self.tracker_epoch = self.current_epoch

        self.get_predictions(batch)

        return True

    def plot(self, stage: str):
        if not self.validate_plot():
            return False

        hep.style.use(hep.style.ATLAS)

        plot_ratio_distributions(
            self.dists,
            save_dir=self.plotting_dirs["dist"],
            save_prefix=f"{stage}_{self.current_epoch}_",
            atlas_marker=self.plotting_conf.get("atlas_label", "Internal"),
        )

        self.log_artifacts()

        return True

    def reset(self) -> None:
        self.dists = {}
