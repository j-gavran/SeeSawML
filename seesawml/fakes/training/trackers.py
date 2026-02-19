from __future__ import annotations

import copy
import os

import hist
import mplhep as hep
import numpy as np
from f9columnar.ml.dataloader_helpers import DatasetColumn, column_selection_from_dict
from f9columnar.ml.hdf5_dataloader import WeightedBatchType
from f9columnar.utils.helpers import load_json
from omegaconf import DictConfig

from seesawml.fakes.models.loss import DensityRatio
from seesawml.fakes.training.tracker_plotting import (
    plot_num_den_hists,
    plot_num_den_hists_with_errors,
    plot_num_den_weights,
    plot_num_den_weights_with_errors,
    plot_ratio_distributions,
)
from seesawml.fakes.utils import get_num_den_weights, get_num_den_weights_with_errors
from seesawml.models.ensembles import torch_predict_from_ensemble_logits
from seesawml.models.tracker import Tracker
from seesawml.utils.helpers import get_log_binning, to_cpu_numpy


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
        if name.endswith("_up_errors") or name.endswith("_down_errors"):
            name = "_".join(name.split("_")[:-2])
        elif name.endswith("_errors"):
            name = "_".join(name.split("_")[:-1])

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

        if self.module.has_ensemble:
            weights_hs += [
                "data_sub_errors",
                "mc_sub_errors",
                "data_reweighted_up_errors",
                "mc_reweighted_up_errors",
                "data_reweighted_down_errors",
                "mc_reweighted_down_errors",
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

                if self.module.has_ensemble:
                    self.hists[f"{plot_column}_up"][k] = copy.deepcopy(h)
                    self.hists[f"{plot_column}_down"][k] = copy.deepcopy(h)

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
            if self.module.has_ensemble:
                self.hists[f"{plot_column}_up"], self.hists[f"{plot_column}_down"] = {}, {}
            self._data_mc_reweighted(plot_column, column_names)

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {"hists": f"{self.base_dir}/hists/"}

    def _get_predictions(self, batch: WeightedBatchType) -> None:
        X, y, w, _, reports = batch

        data_label, mc_label = 1, 0

        data_mask, mc_mask = y == data_label, y == mc_label

        X_data, X_mc = X[data_mask], X[mc_mask]
        w_data, w_mc = w[data_mask], w[mc_mask]

        # get reweighted weights and densities for data and mc
        w_nn_data, density_data, f_data = get_num_den_weights(
            self.module, X_data, is_data=True, density_ratio=self.density_ratio
        )
        w_nn_mc, density_mc, f_mc = get_num_den_weights(
            self.module, X_mc, is_data=False, density_ratio=self.density_ratio
        )

        if self.events_column is None:
            raise RuntimeError("Events column not loaded!")

        numer_idx = self.events_column.offset_numer_columns_idx
        X_data, X_mc = X_data[:, numer_idx], X_mc[:, numer_idx]

        # flatten tensors
        w_nn_data, w_nn_mc = w_nn_data.flatten(), w_nn_mc.flatten()
        density_mc, density_data = density_mc.flatten(), density_data.flatten()
        f_data, f_mc = f_data.flatten(), f_mc.flatten()

        # convert to numpy
        X_data_np, X_mc_np, w_data_np, w_mc_np = to_cpu_numpy(X_data, X_mc, w_data, w_mc)
        w_nn_data_np, w_nn_mc_np = to_cpu_numpy(w_nn_data, w_nn_mc)
        density_data_np, density_mc_np = to_cpu_numpy(density_data, density_mc)
        f_data_np, f_mc_np = to_cpu_numpy(f_data, f_mc)

        # handle NaN values
        w_nn_data_np, w_nn_mc_np = np.nan_to_num(w_nn_data_np), np.nan_to_num(w_nn_mc_np)

        # inverse transform features for plotting if scaler is provided
        feature_scaler = reports["numer_scaler"]
        if feature_scaler is not None:
            if X_data_np.shape[0] >= 1:
                X_data_np = feature_scaler.inverse_transform(X_data_np)

            if X_mc_np.shape[0] >= 1:
                X_mc_np = feature_scaler.inverse_transform(X_mc_np)

        # precompute reweighted weight arrays
        w_data_reweighted = w_data_np * w_nn_data_np
        w_mc_reweighted = w_mc_np * w_nn_mc_np

        # fill weight histograms
        self.weights_hists["data_prescales"].fill(w_data_np)
        self.weights_hists["mc_weights"].fill(w_mc_np)
        self.weights_hists["data_out"].fill(f_data_np)
        self.weights_hists["mc_out"].fill(f_mc_np)
        self.weights_hists["data_density"].fill(density_data_np)
        self.weights_hists["mc_density"].fill(density_mc_np)
        self.weights_hists["data_sub"].fill(w_nn_data_np)
        self.weights_hists["mc_sub"].fill(w_nn_mc_np)
        self.weights_hists["data_reweighted"].fill(w_data_reweighted)
        self.weights_hists["mc_reweighted"].fill(w_mc_reweighted)

        # fill variable histograms
        for plot_column, plot_column_dct in self.hists.items():
            for k, h in plot_column_dct.items():
                h_info = self.hists_info[plot_column][k]

                idx = h_info["idx"]

                if h_info["is_reweighted"]:
                    if h_info["is_data"]:
                        h.fill(X_data_np[:, idx], weight=w_data_reweighted)
                    else:
                        h.fill(X_mc_np[:, idx], weight=w_mc_reweighted)
                else:
                    if h_info["is_data"]:
                        h.fill(X_data_np[:, idx], weight=w_data_np)
                    else:
                        h.fill(X_mc_np[:, idx], weight=w_mc_np)

    def _get_predictions_with_errors(self, batch: WeightedBatchType) -> None:
        X, y, w, _, reports = batch

        data_label, mc_label = 1, 0

        data_mask, mc_mask = y == data_label, y == mc_label

        X_data, X_mc = X[data_mask], X[mc_mask]
        w_data, w_mc = w[data_mask], w[mc_mask]

        # get reweighted weights and densities for data and mc (with errors)
        w_nn_data_mean, w_nn_data_std, density_data_mean, _, f_data_mean, _ = get_num_den_weights_with_errors(
            self.module, X_data, is_data=True, density_ratio=self.density_ratio
        )
        w_nn_mc_mean, w_nn_mc_std, density_mc_mean, _, f_mc_mean, _ = get_num_den_weights_with_errors(
            self.module, X_mc, is_data=False, density_ratio=self.density_ratio
        )

        if self.events_column is None:
            raise RuntimeError("Events column not loaded!")

        numer_idx = self.events_column.offset_numer_columns_idx
        X_data, X_mc = X_data[:, numer_idx], X_mc[:, numer_idx]

        # flatten tensors
        w_nn_data_mean, w_nn_data_std = w_nn_data_mean.flatten(), w_nn_data_std.flatten()
        w_nn_mc_mean, w_nn_mc_std = w_nn_mc_mean.flatten(), w_nn_mc_std.flatten()
        density_data_mean, density_mc_mean = density_data_mean.flatten(), density_mc_mean.flatten()
        f_data_mean, f_mc_mean = f_data_mean.flatten(), f_mc_mean.flatten()

        # convert to numpy
        X_data_np, X_mc_np, w_data_np, w_mc_np = to_cpu_numpy(X_data, X_mc, w_data, w_mc)
        w_nn_data_mean_np, w_nn_data_std_np = to_cpu_numpy(w_nn_data_mean, w_nn_data_std)
        w_nn_mc_mean_np, w_nn_mc_std_np = to_cpu_numpy(w_nn_mc_mean, w_nn_mc_std)
        density_data_mean_np, density_mc_mean_np = to_cpu_numpy(density_data_mean, density_mc_mean)
        f_data_mean_np, f_mc_mean_np = to_cpu_numpy(f_data_mean, f_mc_mean)

        # handle NaN values
        w_nn_data_mean_np, w_nn_mc_mean_np = np.nan_to_num(w_nn_data_mean_np), np.nan_to_num(w_nn_mc_mean_np)
        w_nn_data_std_np, w_nn_mc_std_np = np.nan_to_num(w_nn_data_std_np), np.nan_to_num(w_nn_mc_std_np)

        # inverse transform features for plotting if scaler is provided
        feature_scaler = reports["numer_scaler"]
        if feature_scaler is not None:
            if X_data_np.shape[0] >= 1:
                X_data_np = feature_scaler.inverse_transform(X_data_np)

            if X_mc_np.shape[0] >= 1:
                X_mc_np = feature_scaler.inverse_transform(X_mc_np)

        # precompute reweighted weight arrays (each used in both weight and variable hists)
        w_data_reweighted = w_data_np * w_nn_data_mean_np
        w_mc_reweighted = w_mc_np * w_nn_mc_mean_np
        w_data_reweighted_up = w_data_np * (w_nn_data_mean_np + w_nn_data_std_np)
        w_data_reweighted_down = w_data_np * (w_nn_data_mean_np - w_nn_data_std_np)
        w_mc_reweighted_up = w_mc_np * (w_nn_mc_mean_np + w_nn_mc_std_np)
        w_mc_reweighted_down = w_mc_np * (w_nn_mc_mean_np - w_nn_mc_std_np)

        # fill weight histograms
        self.weights_hists["data_prescales"].fill(w_data_np)
        self.weights_hists["mc_weights"].fill(w_mc_np)
        self.weights_hists["data_out"].fill(f_data_mean_np)
        self.weights_hists["mc_out"].fill(f_mc_mean_np)
        self.weights_hists["data_density"].fill(density_data_mean_np)
        self.weights_hists["mc_density"].fill(density_mc_mean_np)
        self.weights_hists["data_sub"].fill(w_nn_data_mean_np)
        self.weights_hists["mc_sub"].fill(w_nn_mc_mean_np)
        self.weights_hists["data_reweighted"].fill(w_data_reweighted)
        self.weights_hists["mc_reweighted"].fill(w_mc_reweighted)

        # fill weight histograms (errors)
        self.weights_hists["data_sub_errors"].fill(w_nn_data_std_np)
        self.weights_hists["mc_sub_errors"].fill(w_nn_mc_std_np)
        self.weights_hists["data_reweighted_up_errors"].fill(w_data_reweighted_up)
        self.weights_hists["mc_reweighted_up_errors"].fill(w_mc_reweighted_up)
        self.weights_hists["data_reweighted_down_errors"].fill(w_data_reweighted_down)
        self.weights_hists["mc_reweighted_down_errors"].fill(w_mc_reweighted_down)

        # fill variable histograms (nominal + up/down)
        for plot_column, plot_column_info in self.hists_info.items():
            for k, h_info in plot_column_info.items():
                idx = h_info["idx"]

                h = self.hists[plot_column][k]
                h_up = self.hists[f"{plot_column}_up"][k]
                h_down = self.hists[f"{plot_column}_down"][k]

                if h_info["is_reweighted"]:
                    if h_info["is_data"]:
                        h.fill(X_data_np[:, idx], weight=w_data_reweighted)
                        h_up.fill(X_data_np[:, idx], weight=w_data_reweighted_up)
                        h_down.fill(X_data_np[:, idx], weight=w_data_reweighted_down)
                    else:
                        h.fill(X_mc_np[:, idx], weight=w_mc_reweighted)
                        h_up.fill(X_mc_np[:, idx], weight=w_mc_reweighted_up)
                        h_down.fill(X_mc_np[:, idx], weight=w_mc_reweighted_down)
                else:
                    if h_info["is_data"]:
                        h.fill(X_data_np[:, idx], weight=w_data_np)
                        h_up.fill(X_data_np[:, idx], weight=w_data_np)
                        h_down.fill(X_data_np[:, idx], weight=w_data_np)
                    else:
                        h.fill(X_mc_np[:, idx], weight=w_mc_np)
                        h_up.fill(X_mc_np[:, idx], weight=w_mc_np)
                        h_down.fill(X_mc_np[:, idx], weight=w_mc_np)

    def get_binned_chi2(self) -> tuple[float, float]:
        chi2_data_lst, chi2_mc_list = [], []

        nominal_keys = [k for k in self.hists if not k.endswith("_up") and not k.endswith("_down")]

        for k in nominal_keys:
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

        if self.module.has_ensemble:
            self._get_predictions_with_errors(batch)
        else:
            self._get_predictions(batch)

        return True

    def plot(self, stage: str) -> bool:
        if not self.validate_plot():
            return False

        hep.style.use(hep.style.ATLAS)

        plot_weights = plot_num_den_weights_with_errors if self.module.has_ensemble else plot_num_den_weights
        plot_hists = plot_num_den_hists_with_errors if self.module.has_ensemble else plot_num_den_hists

        plot_weights(
            self.weights_hists,
            save_dir=self.plotting_dirs["hists"],
            save_prefix=f"{self.num_den}_{stage}_{self.current_epoch}_",
            atlas_marker=self.plotting_conf.subtraction_plot.get("atlas_label", "Internal"),
        )
        plot_hists(
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

        if self.module.has_ensemble:
            self.dists["logits_up"] = copy.deepcopy(self.dists["logits"])
            self.dists["logits_down"] = copy.deepcopy(self.dists["logits"])

            self.dists["density_up"] = copy.deepcopy(self.dists["density"])
            self.dists["density_down"] = copy.deepcopy(self.dists["density"])

    def _get_predictions(self, batch: WeightedBatchType) -> None:
        X = batch[0]

        y_hat = self.module(X).flatten()
        y_ratio = self.density_ratio(y_hat)

        self.dists["density"].fill(y_ratio.cpu().numpy())
        self.dists["logits"].fill(y_hat.cpu().numpy())

    def _get_predictions_with_errors(self, batch: WeightedBatchType) -> None:
        X = batch[0]

        model_output = self.module(X)
        y_hat_mean, y_hat_std = torch_predict_from_ensemble_logits(model_output)

        y_ratio_mean, y_ratio_std = self.density_ratio.ratio_with_errors(y_hat_mean, y_hat_std)

        y_hat_mean_np, y_hat_std_np = to_cpu_numpy(y_hat_mean, y_hat_std)
        y_ratio_mean_np, y_ratio_std_np = to_cpu_numpy(y_ratio_mean, y_ratio_std)

        self.dists["density"].fill(y_ratio_mean_np)
        self.dists["logits"].fill(y_hat_mean_np)

        self.dists["density_up"].fill(y_ratio_mean_np + y_ratio_std_np)
        self.dists["density_down"].fill(y_ratio_mean_np - y_ratio_std_np)

        self.dists["logits_up"].fill(y_hat_mean_np + y_hat_std_np)
        self.dists["logits_down"].fill(y_hat_mean_np - y_hat_std_np)

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

        if self.module.has_ensemble:
            self._get_predictions_with_errors(batch)
        else:
            self._get_predictions(batch)

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
