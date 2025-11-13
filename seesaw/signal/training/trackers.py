import logging
from typing import Any

import mplhep as hep
import numpy as np
import torch
from f9columnar.ml.hdf5_dataloader import FullWeightedBatchType, WeightedBatchType
from omegaconf import DictConfig
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryROC,
    MulticlassConfusionMatrix,
)

from seesaw.models.tracker import Tracker
from seesaw.signal.training.group_plotting import (
    plot_group_one_vs_rest_discriminant,
    plot_group_one_vs_rest_roc,
    plot_group_one_vs_rest_score,
    plot_multiclass_group_discriminant,
    plot_multiclass_group_score,
)
from seesaw.signal.training.tracker_plotting import (
    plot_binary_bkg_rej_vs_sig_eff,
    plot_binary_confusion_matrix,
    plot_binary_model_score,
    plot_binary_precision_recall,
    plot_binary_roc,
    plot_multiclass_confusion_matrix,
    plot_multiclass_discriminant,
    plot_multiclass_discriminant_one_vs_rest,
    plot_multiclass_model_score,
    plot_multiclass_one_vs_rest_roc,
    plot_multiclass_one_vs_rest_score,
    plot_multiclass_tsne_pca,
)


class SigBkgClassifierTracker(Tracker):
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

        self._max_events = 10**7

        self.sig_label: int
        self.bkg_label: int

        self.roc_metric = BinaryROC(thresholds=None)
        self.prc_metric = BinaryPrecisionRecallCurve(thresholds=None)
        self.auroc_metric = BinaryAUROC(thresholds=None)
        self.auprc_metric = BinaryAveragePrecision(thresholds=None)
        self.bcm = BinaryConfusionMatrix()

        self.accumulated_true: list[torch.Tensor] = []
        self.accumulated_pred: list[torch.Tensor] = []
        self.accumulated_logit_pred: list[torch.Tensor] = []
        self.accumulated_mc_weights: list[torch.Tensor] = []

        self.current_events = 0

    def reset(self) -> None:
        self.roc_metric.reset()
        self.prc_metric.reset()
        self.auroc_metric.reset()
        self.auprc_metric.reset()
        self.bcm.reset()

        self.accumulated_true.clear()
        self.accumulated_pred.clear()
        self.accumulated_logit_pred.clear()
        self.accumulated_mc_weights.clear()

        self.current_events = 0

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {
            "roc": f"{self.base_dir}/roc/",
            "confmat": f"{self.base_dir}/confmat/",
            "scores": f"{self.base_dir}/score/",
            "custom_scores": f"{self.base_dir}/custom_score/",
        }

    def calculate_roc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = self.roc_metric(y_pred, y_true)
        fpr, tpr = fpr.numpy(), tpr.numpy()

        return fpr, tpr

    def calulate_prc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        precision, recall, _ = self.prc_metric(y_pred, y_true)
        precision, recall = precision.numpy(), recall.numpy()

        return precision, recall

    def calculate_bcm(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
        bcm = self.bcm(y_pred, y_true)
        bcm = bcm.numpy()

        return bcm

    def compute(self, batch: WeightedBatchType, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if not self.validate_compute():
            return False

        if self.current_events >= self._max_events:
            return False

        X, y_true, mc_weights, _, reports = batch

        if "signal" not in reports["class_labels"] or "background" not in reports["class_labels"]:
            logging.warning("Expected 'signal' and 'background' labels in class_labels, but they are missing.")
            return False

        self.sig_label = reports["class_labels"]["signal"]
        self.bkg_label = reports["class_labels"]["background"]

        y_pred_logits = self.module(X).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)

        y_true = y_true.long()

        self.accumulated_true.append(y_true.cpu())
        self.accumulated_pred.append(y_pred.cpu())
        self.accumulated_logit_pred.append(y_pred_logits.cpu())
        self.accumulated_mc_weights.append(mc_weights.cpu())

        self.auroc_metric.update(y_pred, y_true)
        self.auprc_metric.update(y_pred, y_true)

        self.current_events += X.shape[0]

        return True

    def plot(self, stage: str) -> bool:
        if self.validate_compute():
            auroc = self.auroc_metric.compute().item()
            auprc = self.auprc_metric.compute().item()

            self.module.log(f"{stage}_auroc", auroc)
            self.module.log(f"{stage}_auprc", auprc)

        if not self.validate_plot():
            self.reset()
            return False

        hep.style.use(hep.style.ATLAS)

        accumulated_true = torch.cat(self.accumulated_true)
        accumulated_pred = torch.cat(self.accumulated_pred)
        accumulated_logit_pred = torch.cat(self.accumulated_logit_pred)
        accumulated_mc_weights = torch.cat(self.accumulated_mc_weights)

        fpr, tpr = self.calculate_roc(accumulated_pred, accumulated_true)
        precision, recall = self.calulate_prc(accumulated_pred, accumulated_true)
        bcm = self.calculate_bcm(accumulated_pred, accumulated_true)

        if stage == "test":
            save_postfix = "test_epoch"
        else:
            save_postfix = f"{stage}_epoch_{self.current_epoch}"

        plot_binary_roc(
            fpr,
            tpr,
            auroc,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_binary_bkg_rej_vs_sig_eff(
            fpr,
            tpr,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_binary_precision_recall(
            precision,
            recall,
            auprc,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_binary_confusion_matrix(
            bcm,
            self.sig_label,
            self.bkg_label,
            save_path=self.plotting_dirs["confmat"],
            save_postfix=save_postfix,
        )

        accumulated_true_np = accumulated_true.to(torch.float32).numpy()
        accumulated_pred_np = accumulated_pred.to(torch.float32).numpy()
        accumulated_logit_pred_np = accumulated_logit_pred.to(torch.float32).numpy()
        accumulated_mc_weights_np = accumulated_mc_weights.to(torch.float32).numpy()

        model_scores = {
            "normalised_weighted": (True, True),
            "normalised_unweighted": (True, False),
            "unnormalised_weighted": (False, True),
            "unnormalised_unweighted": (False, False),
        }

        sigmoid_plot_conf = self.plotting_conf.score_plot.sigmoid_plot
        logit_plot_conf = self.plotting_conf.score_plot.logit_plot

        for use_sigmoid in [True, False]:
            if use_sigmoid:
                score, plot_conf = accumulated_pred_np, sigmoid_plot_conf
            else:
                score, plot_conf = accumulated_logit_pred_np, logit_plot_conf

            for score_type, (use_density, use_mc_weights) in model_scores.items():
                if not plot_conf[score_type]:
                    continue

                _save_postfix = f"{score_type}_{save_postfix}"

                if use_sigmoid:
                    _save_postfix += "_sigmoid"

                plot_binary_model_score(
                    score,
                    accumulated_true_np,
                    self.sig_label,
                    self.bkg_label,
                    n_bins=plot_conf.n_bins,
                    density=use_density,
                    mc_weights=accumulated_mc_weights_np if use_mc_weights else None,
                    save_path=self.plotting_dirs["scores"],
                    save_postfix=_save_postfix,
                )

        self.log_artifacts()
        self.reset()

        return True


class SigBkgMulticlassClassifierTracker(Tracker):
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

        self._max_events = 10**7

        self.num_classes = len(self.dataset_conf.classes)

        self.class_labels: dict[str, int] | None = None

        self.accumulated_true: list[torch.Tensor] = []
        self.accumulated_pred: list[torch.Tensor] = []
        self.accumulated_logits: list[torch.Tensor] = []

        self.tsne_pca_X: list[torch.Tensor] = []
        self.tsne_pca_y_true: list[torch.Tensor] = []
        self.tsne_pca_y_pred: list[torch.Tensor] = []

        self.current_events = 0

    def _set_class_labels(self, reports: dict[str, Any]) -> None:
        self.class_labels = reports.get("class_labels", None)

        if self.class_labels is None:
            raise RuntimeError("Class labels are not provided in the reports.")

    def reset(self) -> None:
        self.accumulated_true.clear()
        self.accumulated_pred.clear()
        self.accumulated_logits.clear()

        self.tsne_pca_X.clear()
        self.tsne_pca_y_true.clear()
        self.tsne_pca_y_pred.clear()

        self.current_events = 0

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {
            "roc": f"{self.base_dir}/roc/",
            "confmat": f"{self.base_dir}/confmat/",
            "scores": f"{self.base_dir}/score/",
            "custom_scores": f"{self.base_dir}/custom_score/",
        }

    def calculate_cm(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        cm = MulticlassConfusionMatrix(self.num_classes)
        return cm(torch.argmax(y_pred, dim=1), torch.argmax(y_true, dim=1)).numpy()

    def compute(self, batch: WeightedBatchType, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if (self.current_epoch == 0 and self.current_events == 0) or self.stage == "test" or self.class_labels is None:
            self._set_class_labels(batch[-1])

        if not self.validate_compute():
            return False

        if self.current_events >= self._max_events:
            return False

        X, y_true, _, _, _ = batch
        y_true = y_true.long().cpu()

        y_pred_logits = self.module(X)
        y_pred = torch.softmax(y_pred_logits, dim=1).cpu()

        if self.current_events == 0:
            self.tsne_pca_X.append(X.cpu())
            self.tsne_pca_y_pred.append(torch.argmax(y_pred, dim=1))
            self.tsne_pca_y_true.append(torch.argmax(y_true, dim=1))

        self.accumulated_logits.append(y_pred_logits.cpu())
        self.accumulated_true.append(y_true)
        self.accumulated_pred.append(y_pred)

        self.current_events += X.shape[0]

        return True

    def plot(self, stage: str) -> bool:
        if not self.validate_plot():
            return False

        hep.style.use(hep.style.ATLAS)

        accumulated_true = torch.cat(self.accumulated_true)
        accumulated_pred = torch.cat(self.accumulated_pred)
        np_accumulated_logits = torch.cat(self.accumulated_logits).numpy()

        if not self.plotting_conf.get("disable_projections", True):
            tsne_pca_X = torch.cat(self.tsne_pca_X).numpy()
            tsne_pca_y_true = torch.cat(self.tsne_pca_y_true).numpy()
            tsne_pca_y_pred = torch.cat(self.tsne_pca_y_pred).numpy()

        cm = self.calculate_cm(accumulated_true, accumulated_pred)

        np_accumulated_true = accumulated_true.numpy()
        np_accumulated_pred = accumulated_pred.numpy()

        if stage == "test":
            save_postfix = "test_epoch"
        else:
            save_postfix = f"{stage}_epoch_{self.current_epoch}"

        plot_multiclass_confusion_matrix(
            cm,
            self.class_labels,
            save_path=self.plotting_dirs["confmat"],
            save_postfix=save_postfix,
        )
        plot_multiclass_model_score(
            np_accumulated_pred,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
            is_softmax=True,
        )
        plot_multiclass_model_score(
            np_accumulated_logits,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
            is_softmax=False,
        )
        if not self.plotting_conf.get("disable_projections", True):
            for pca_tsne in [True, False]:
                plot_multiclass_tsne_pca(
                    tsne_pca_X,
                    tsne_pca_y_pred,
                    tsne_pca_y_true,
                    self.class_labels,
                    save_path=self.plotting_dirs["scores"],
                    save_postfix=save_postfix,
                    use_pca=pca_tsne,
                )
        # Always produce per-class ROC curves
        plot_multiclass_one_vs_rest_roc(
            np_accumulated_pred,
            np_accumulated_true,
            self.class_labels,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_multiclass_discriminant(
            np_accumulated_pred,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
        )
        plot_multiclass_one_vs_rest_score(
            np_accumulated_pred,
            np_accumulated_true,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
        )
        plot_multiclass_discriminant_one_vs_rest(
            np_accumulated_pred,
            np_accumulated_true,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
        )

        # Optional grouped outputs when dataset defines groups
        custom_groups = None
        if "custom_groups" in self.dataset_conf:
            custom_groups = self.dataset_conf.custom_groups
        elif "scores" in self.dataset_conf:
            custom_groups = self.dataset_conf.scores

        if custom_groups:
            if self.class_labels is None:
                raise RuntimeError("Class labels are not set; cannot compute custom group plots.")
            name_to_idx: dict[str, int] = self.class_labels
            group_indices: dict[str, list[int]] = {}
            for gname, members in dict(custom_groups).items():
                idxs = [name_to_idx[m] for m in list(members) if m in name_to_idx]
                if len(idxs) > 0:
                    group_indices[str(gname)] = idxs

            if len(group_indices) > 0:
                plot_multiclass_group_score(
                    np_accumulated_pred,
                    np_accumulated_true,
                    group_indices,
                    save_path=self.plotting_dirs["custom_scores"],
                    save_postfix=save_postfix,
                )
                plot_multiclass_group_discriminant(
                    np_accumulated_pred,
                    np_accumulated_true,
                    group_indices,
                    save_path=self.plotting_dirs["custom_scores"],
                    save_postfix=save_postfix,
                )

        # Optional per-group OVR from plotting config
        if "custom_groups" in self.plotting_conf:
            for gname, members in self.plotting_conf.custom_groups.items():
                members_list = list(members)
                plot_group_one_vs_rest_score(
                    np_accumulated_pred,
                    np_accumulated_true,
                    self.class_labels,
                    gname,
                    members_list,
                    save_path=self.plotting_dirs["scores"],
                    save_postfix=save_postfix,
                )
                plot_group_one_vs_rest_roc(
                    np_accumulated_pred,
                    np_accumulated_true,
                    self.class_labels,
                    gname,
                    members_list,
                    save_path=self.plotting_dirs["roc"],
                    save_postfix=save_postfix,
                )
                plot_group_one_vs_rest_discriminant(
                    np_accumulated_pred,
                    np_accumulated_true,
                    self.class_labels,
                    gname,
                    members_list,
                    save_path=self.plotting_dirs["scores"],
                    save_postfix=save_postfix,
                )

        self.log_artifacts()
        self.reset()

        return True


class JaggedSigBkgClassifierTracker(Tracker):
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

        self._max_events = 10**7

        self.sig_label: int
        self.bkg_label: int

        self.roc_metric = BinaryROC(thresholds=None)
        self.prc_metric = BinaryPrecisionRecallCurve(thresholds=None)
        self.auroc_metric = BinaryAUROC(thresholds=None)
        self.auprc_metric = BinaryAveragePrecision(thresholds=None)
        self.bcm = BinaryConfusionMatrix()

        self.accumulated_true: list[torch.Tensor] = []
        self.accumulated_pred: list[torch.Tensor] = []
        self.accumulated_logit_pred: list[torch.Tensor] = []
        self.accumulated_mc_weights: list[torch.Tensor] = []

        self.current_events = 0

    def reset(self) -> None:
        self.roc_metric.reset()
        self.prc_metric.reset()
        self.auroc_metric.reset()
        self.auprc_metric.reset()
        self.bcm.reset()

        self.accumulated_true.clear()
        self.accumulated_pred.clear()
        self.accumulated_logit_pred.clear()
        self.accumulated_mc_weights.clear()

        self.current_events = 0

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {
            "roc": f"{self.base_dir}/roc/",
            "confmat": f"{self.base_dir}/confmat/",
            "scores": f"{self.base_dir}/score/",
        }

    def calculate_roc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = self.roc_metric(y_pred, y_true)
        fpr, tpr = fpr.numpy(), tpr.numpy()

        return fpr, tpr

    def calulate_prc(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        precision, recall, _ = self.prc_metric(y_pred, y_true)
        precision, recall = precision.numpy(), recall.numpy()

        return precision, recall

    def calculate_bcm(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
        bcm = self.bcm(y_pred, y_true)
        bcm = bcm.numpy()

        return bcm

    def compute(self, batch: FullWeightedBatchType, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if not self.validate_compute():
            return False

        if self.current_events >= self._max_events:
            return False

        reports = batch[-1]
        Xs: list[torch.Tensor] = []

        for k in batch[0].keys():
            if k != "events":
                Xs.append(batch[0][k][0])

        X_events = batch[0]["events"][0]
        y_true, mc_weights, y_classes = batch[0]["events"][1], batch[0]["events"][2], batch[0]["events"][3]

        if y_true is None or mc_weights is None or y_classes is None:
            raise ValueError("y, w, or y_classes is None!")

        if "signal" not in reports["class_labels"] or "background" not in reports["class_labels"]:
            logging.warning("Expected 'signal' and 'background' labels in class_labels, but they are missing.")
            return False

        self.sig_label = reports["class_labels"]["signal"]
        self.bkg_label = reports["class_labels"]["background"]

        y_pred_logits = self.module(X_events, Xs).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)

        y_true = y_true.long()

        self.accumulated_true.append(y_true.cpu())
        self.accumulated_pred.append(y_pred.cpu())
        self.accumulated_logit_pred.append(y_pred_logits.cpu())
        self.accumulated_mc_weights.append(mc_weights.cpu())

        self.auroc_metric.update(y_pred, y_true)
        self.auprc_metric.update(y_pred, y_true)

        self.current_events += X_events.shape[0]

        return True

    def plot(self, stage: str) -> bool:
        if self.validate_compute():
            auroc = self.auroc_metric.compute().item()
            auprc = self.auprc_metric.compute().item()

            self.module.log(f"{stage}_auroc", auroc)
            self.module.log(f"{stage}_auprc", auprc)

        if not self.validate_plot():
            self.reset()
            return False

        hep.style.use(hep.style.ATLAS)

        accumulated_true = torch.cat(self.accumulated_true)
        accumulated_pred = torch.cat(self.accumulated_pred)
        accumulated_logit_pred = torch.cat(self.accumulated_logit_pred)
        accumulated_mc_weights = torch.cat(self.accumulated_mc_weights)

        fpr, tpr = self.calculate_roc(accumulated_pred, accumulated_true)
        precision, recall = self.calulate_prc(accumulated_pred, accumulated_true)
        bcm = self.calculate_bcm(accumulated_pred, accumulated_true)

        if stage == "test":
            save_postfix = "test_epoch"
        else:
            save_postfix = f"{stage}_epoch_{self.current_epoch}"

        plot_binary_roc(
            fpr,
            tpr,
            auroc,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_binary_bkg_rej_vs_sig_eff(
            fpr,
            tpr,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_binary_precision_recall(
            precision,
            recall,
            auprc,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_binary_confusion_matrix(
            bcm,
            self.sig_label,
            self.bkg_label,
            save_path=self.plotting_dirs["confmat"],
            save_postfix=save_postfix,
        )

        accumulated_true_np = accumulated_true.to(torch.float32).numpy()
        accumulated_pred_np = accumulated_pred.to(torch.float32).numpy()
        accumulated_logit_pred_np = accumulated_logit_pred.to(torch.float32).numpy()
        accumulated_mc_weights_np = accumulated_mc_weights.to(torch.float32).numpy()

        model_scores = {
            "normalised_weighted": (True, True),
            "normalised_unweighted": (True, False),
            "unnormalised_weighted": (False, True),
            "unnormalised_unweighted": (False, False),
        }

        sigmoid_plot_conf = self.plotting_conf.score_plot.sigmoid_plot
        logit_plot_conf = self.plotting_conf.score_plot.logit_plot

        for use_sigmoid in [True, False]:
            if use_sigmoid:
                score, plot_conf = accumulated_pred_np, sigmoid_plot_conf
            else:
                score, plot_conf = accumulated_logit_pred_np, logit_plot_conf

            for score_type, (use_density, use_mc_weights) in model_scores.items():
                if not plot_conf[score_type]:
                    continue

                _save_postfix = f"{score_type}_{save_postfix}"

                if use_sigmoid:
                    _save_postfix += "_sigmoid"

                plot_binary_model_score(
                    score,
                    accumulated_true_np,
                    self.sig_label,
                    self.bkg_label,
                    n_bins=plot_conf.n_bins,
                    density=use_density,
                    mc_weights=accumulated_mc_weights_np if use_mc_weights else None,
                    save_path=self.plotting_dirs["scores"],
                    save_postfix=_save_postfix,
                )

        self.log_artifacts()
        self.reset()

        return True


class JaggedSigBkgMulticlassClassifierTracker(Tracker):
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

        self._max_events = 10**7

        self.num_classes = len(self.dataset_conf.classes)

        self.class_labels: dict[str, int] | None = None

        self.accumulated_true: list[torch.Tensor] = []
        self.accumulated_pred: list[torch.Tensor] = []
        self.accumulated_logits: list[torch.Tensor] = []

        self.tsne_pca_X: list[torch.Tensor] = []
        self.tsne_pca_y_true: list[torch.Tensor] = []
        self.tsne_pca_y_pred: list[torch.Tensor] = []

        self.current_events = 0

    def _set_class_labels(self, reports: dict[str, Any]) -> None:
        self.class_labels = reports.get("class_labels", None)
        if self.class_labels is None:
            raise RuntimeError("Class labels are not provided in the reports.")

    def reset(self) -> None:
        self.accumulated_true.clear()
        self.accumulated_pred.clear()
        self.accumulated_logits.clear()

        self.tsne_pca_X.clear()
        self.tsne_pca_y_true.clear()
        self.tsne_pca_y_pred.clear()

        self.current_events = 0

    @property
    def plotting_dirs(self) -> dict[str, str]:
        return {
            "roc": f"{self.base_dir}/roc/",
            "confmat": f"{self.base_dir}/confmat/",
            "scores": f"{self.base_dir}/score/",
        }

    def calculate_cm(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        cm = MulticlassConfusionMatrix(self.num_classes)
        return cm(torch.argmax(y_pred, dim=1), torch.argmax(y_true, dim=1)).numpy()

    def compute(self, batch: FullWeightedBatchType, stage: str) -> bool:
        self.stage, self.current_epoch = stage, self.module.current_epoch

        if not self.validate_compute():
            return False

        if self.current_events >= self._max_events:
            return False

        reports = batch[-1]
        if self.class_labels is None:
            self._set_class_labels(reports)

        Xs: list[torch.Tensor] = []
        for k in batch[0].keys():
            if k != "events":
                Xs.append(batch[0][k][0])

        X = batch[0]["events"][0]
        y_true = batch[0]["events"][1]
        if y_true is None:
            return False

        y_true = y_true.long().cpu()

        y_pred_logits = self.module(X, Xs)
        y_pred = torch.softmax(y_pred_logits, dim=1).cpu()

        if self.current_events == 0:
            self.tsne_pca_X.append(X.cpu())
            self.tsne_pca_y_pred.append(torch.argmax(y_pred, dim=1))
            self.tsne_pca_y_true.append(torch.argmax(y_true, dim=1))

        self.accumulated_logits.append(y_pred_logits.cpu())
        self.accumulated_true.append(y_true)
        self.accumulated_pred.append(y_pred)

        self.current_events += X.shape[0]

        return True

    def plot(self, stage: str) -> bool:
        if not self.validate_plot():
            return False

        if self.class_labels is None:
            return False

        hep.style.use(hep.style.ATLAS)

        accumulated_true = torch.cat(self.accumulated_true)
        accumulated_pred = torch.cat(self.accumulated_pred)
        np_accumulated_logits = torch.cat(self.accumulated_logits).numpy()

        if not self.plotting_conf.get("disable_projections", True):
            tsne_pca_X = torch.cat(self.tsne_pca_X).numpy() if len(self.tsne_pca_X) != 0 else np.zeros((0,))
            tsne_pca_y_true = (
                torch.cat(self.tsne_pca_y_true).numpy() if len(self.tsne_pca_y_true) != 0 else np.zeros((0,))
            )
            tsne_pca_y_pred = (
                torch.cat(self.tsne_pca_y_pred).numpy() if len(self.tsne_pca_y_pred) != 0 else np.zeros((0,))
            )

        cm = self.calculate_cm(accumulated_true, accumulated_pred)

        np_accumulated_true = accumulated_true.numpy()
        np_accumulated_pred = accumulated_pred.numpy()

        if stage == "test":
            save_postfix = "test_epoch"
        else:
            save_postfix = f"{stage}_epoch_{self.current_epoch}"

        plot_multiclass_confusion_matrix(
            cm,
            self.class_labels,
            save_path=self.plotting_dirs["confmat"],
            save_postfix=save_postfix,
        )
        plot_multiclass_model_score(
            np_accumulated_pred,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
            is_softmax=True,
        )
        plot_multiclass_model_score(
            np_accumulated_logits,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
            is_softmax=False,
        )
        if not self.plotting_conf.get("disable_projections", True):
            for pca_tsne in [True, False]:
                plot_multiclass_tsne_pca(
                    tsne_pca_X,
                    tsne_pca_y_pred,
                    tsne_pca_y_true,
                    self.class_labels,
                    save_path=self.plotting_dirs["scores"],
                    save_postfix=save_postfix,
                    use_pca=pca_tsne,
                )
        # Always produce per-class ROC curves
        plot_multiclass_one_vs_rest_roc(
            np_accumulated_pred,
            np_accumulated_true,
            self.class_labels,
            save_path=self.plotting_dirs["roc"],
            save_postfix=save_postfix,
        )
        plot_multiclass_discriminant(
            np_accumulated_pred,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
        )
        plot_multiclass_one_vs_rest_score(
            np_accumulated_pred,
            np_accumulated_true,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
        )
        plot_multiclass_discriminant_one_vs_rest(
            np_accumulated_pred,
            np_accumulated_true,
            self.class_labels,
            save_path=self.plotting_dirs["scores"],
            save_postfix=save_postfix,
        )

        # Optional grouped outputs when dataset defines groups
        groups_cfg = None
        if "custom_groups" in self.dataset_conf:
            groups_cfg = self.dataset_conf.custom_groups
        elif "scores" in self.dataset_conf:
            groups_cfg = self.dataset_conf.scores

        if not groups_cfg and "custom_groups" in self.plotting_conf:
            groups_cfg = self.plotting_conf.custom_groups

        if groups_cfg:
            name_to_idx = self.class_labels
            group_indices = {}
            for gname, members in dict(groups_cfg).items():
                idxs = [name_to_idx[m] for m in list(members) if m in name_to_idx]
                if len(idxs) > 0:
                    group_indices[str(gname)] = idxs

            if len(group_indices) > 0:
                plot_multiclass_group_score(
                    np_accumulated_pred,
                    np_accumulated_true,
                    group_indices,
                    save_path=self.plotting_dirs["custom_scores"],
                    save_postfix=save_postfix,
                )
                plot_multiclass_group_discriminant(
                    np_accumulated_pred,
                    np_accumulated_true,
                    group_indices,
                    save_path=self.plotting_dirs["custom_scores"],
                    save_postfix=save_postfix,
                )

        self.log_artifacts()
        self.reset()

        return True
