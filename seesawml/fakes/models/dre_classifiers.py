from abc import abstractmethod
from typing import Any

import torch
from f9columnar.ml.hdf5_dataloader import WeightedBatchType
from omegaconf import DictConfig
from torchmetrics.classification import BinaryAccuracy

from seesawml.fakes.models.crack_veto_model import get_crack_veto_model
from seesawml.fakes.models.loss import DensityRatio, DensityRatioLoss
from seesawml.fakes.utils import get_num_den_weights, sample_subtraction_weights
from seesawml.models.ensembles import (
    StackedEnsembleNetWrapper,
    torch_predict_from_ensemble_logits,
)
from seesawml.models.nn_modules import BaseBatchCompletedLightningModule
from seesawml.models.tracker import Tracker
from seesawml.models.utils import build_network


class FakesNNClassifier(BaseBatchCompletedLightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        super().__init__(dataset_conf, model_conf, run_name, tracker)

        self.model_conf = model_conf
        model, self.model_name, selection = build_network(dataset_conf, model_conf, run_name)

        self.has_ensemble = False
        ensemble_config = model_conf.architecture_config.get("ensemble", None)

        self.mc_norm_sigma = self.training_conf.get("mc_norm_sigma", 0.0)
        # Poisson bootstrap: multiply per-event weights by Poisson(1) draws independently
        # per ensemble member during training - this is the Bayesian bootstrap — each member
        # effectively trains on a different resampled dataset
        self.poisson_bootstrap: bool = self.training_conf.get("poisson_bootstrap", False)

        if ensemble_config is not None:
            loss_name = model_conf.training_config.loss
            if loss_name != "bce":
                raise ValueError("Ensemble is only supported for BCE loss!")

            if model_conf.architecture_config.output_dim is None:
                raise ValueError("Backbone output dimension must be specified for ensemble!")

            model = StackedEnsembleNetWrapper(
                model,
                backbone_output_dim=model_conf.architecture_config.output_dim,
                use_log_var=False,
                **ensemble_config,
            )
            self.has_ensemble = True

        self.loss_func = DensityRatioLoss(
            self.training_conf["loss"],
            class_weight=None,
            w_lambda=self.training_conf.get("w_lambda", None),
            ess_lambda=self.training_conf.get("ess_lambda", None),
        )

        if dataset_conf.get("crack_veto", False):
            crack_veto_model = get_crack_veto_model(dataset_conf, selection)

            if model_conf.architecture_config.get("compile", False):
                compile_kwargs = model_conf.architecture_config.get("compile_kwargs", {})
                crack_veto_model.compile(**compile_kwargs)

            self.model = torch.nn.Sequential(crack_veto_model, model)
        else:
            self.model = model

        self.binary_acc = BinaryAccuracy()

    def get_batch_size(self, batch):
        return batch[0].size()[0]

    @abstractmethod
    def get_accuracy(self, y_hat: torch.Tensor, batch: WeightedBatchType) -> torch.Tensor:
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(X)


class NumDenClassifier(FakesNNClassifier):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        super().__init__(dataset_conf, model_conf, run_name, tracker)

    def get_accuracy(self, y_hat: torch.Tensor, batch: WeightedBatchType) -> torch.Tensor:
        return self.binary_acc(y_hat, batch[1])

    def get_loss(self, batch: WeightedBatchType, stage: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        X, y, w, _, reports = batch

        if reports["use_data"] is False or reports["use_mc"] is False:
            raise RuntimeError("Invalid data/mc configuration!")

        y_hat = self(X)

        if not self.has_ensemble:
            y_hat = y_hat.flatten()

        # MC normalization variation: each member sees a different global MC scale,
        # sampled from LogNormal(0, mc_norm_sigma)
        # this encodes the uncertainty on the MC normalization into the ensemble spread
        # mc_norm_sigma = log(1.1) ~ 0.0953 corresponds to a +-10% normalization uncertainty
        if self.has_ensemble and self.training and w is not None and self.mc_norm_sigma > 0.0:
            channels = y_hat.shape[0]
            w = w.unsqueeze(0).expand(channels, -1)
            mc_mask = (y == 0).unsqueeze(0)  # (1, batch); MC events have label 0
            log_scales = torch.randn(channels, 1, device=w.device) * self.mc_norm_sigma
            mc_scales = torch.exp(log_scales).expand(channels, w.shape[1])
            w = torch.where(mc_mask, w * mc_scales, w)

        if self.has_ensemble and self.training and self.poisson_bootstrap:
            channels = y_hat.shape[0]
            if w.dim() == 1:
                w = w.unsqueeze(0).expand(channels, -1)

            w = w * torch.poisson(torch.ones_like(w))

        loss = self.loss_func(y_hat, y, w)

        batch_size = self.get_batch_size(batch)
        stage = "train" if self.training else "val"

        if self.has_ensemble:
            f_mean, f_std = torch_predict_from_ensemble_logits(y_hat)
            r_mean = torch.exp(f_mean)
            r_std = r_mean * f_std
            logits_for_stats = f_mean
            r_for_ess = r_mean
        else:
            logits_for_stats = y_hat
            r_for_ess = torch.exp(y_hat)

        ess = (torch.sum(r_for_ess) ** 2) / (torch.sum(r_for_ess**2) + 1e-9)

        self.log(f"{stage}_ess", ess, batch_size=batch_size)
        self.log(f"{stage}_inv_ess", 1.0 / (ess + 1e-9), batch_size=batch_size)

        self.log(f"{stage}_mean_r", torch.mean(logits_for_stats), batch_size=batch_size)
        self.log(f"{stage}_std_r", torch.std(logits_for_stats), batch_size=batch_size)

        self.log(f"{stage}_max_r", torch.max(logits_for_stats), batch_size=batch_size)
        self.log(f"{stage}_min_r", torch.min(logits_for_stats), batch_size=batch_size)

        if self.has_ensemble:
            self.log(f"{stage}_mean_r_ensemble", torch.mean(r_mean), batch_size=batch_size)
            self.log(f"{stage}_std_r_ensemble", torch.mean(r_std), batch_size=batch_size)

        return loss, y_hat

    def on_validation_epoch_end(self) -> None:
        if self.tracker:
            self.tracker.plot(stage="val")

            chi2_data, chi2_mc = self.tracker.get_binned_chi2()  # type: ignore[attr-defined]
            self.log("val_chi2_data", chi2_data)
            self.log("val_chi2_mc", chi2_mc)
            self.log("val_chi2", chi2_data + chi2_mc)


class RatioClassifier(FakesNNClassifier):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        num_model: torch.nn.Module | None = None,
        den_model: torch.nn.Module | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        super().__init__(dataset_conf, model_conf, run_name, tracker)

        self.density_ratio = DensityRatio(self.training_conf["loss"])

        # quantile threshold for per-channel weight clipping during ratio training.
        self.weight_clip_quantile: float | None = self.training_conf.get("weight_clip_quantile", None)

        if num_model is not None:
            self.num_model = num_model.eval()

        if den_model is not None:
            self.den_model = den_model.eval()

    def state_dict(self) -> dict[str, Any]:  # type: ignore
        return {k: v for k, v in super().state_dict().items() if k.startswith("model.")}

    def get_accuracy(self, y_hat: torch.Tensor, batch: WeightedBatchType) -> torch.Tensor:
        return self.binary_acc(y_hat, batch[3])

    def get_loss(self, batch: WeightedBatchType, stage: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        X, _, w, y_lt, reports = batch

        if reports["use_loose"] is False or reports["use_tight"] is False:
            raise RuntimeError("Invalid loose/tight configuration!")

        if reports["use_data"] is True and reports["use_mc"] is False:
            is_data = True
        elif reports["use_data"] is False and reports["use_mc"] is True:
            is_data = False
        else:
            raise RuntimeError("Invalid data/mc configuration!")

        tight_mask = y_lt == 1.0
        tight_idx = torch.argwhere(tight_mask).flatten()
        tight = X[tight_mask]

        loose_mask = y_lt == 0.0
        loose_idx = torch.argwhere(loose_mask).flatten()
        loose = X[loose_mask]

        y_hat = self(X)

        if self.has_ensemble:
            channels = y_hat.shape[0]

            # per-member subtraction weights (channels, n_tight/loose)

            # during training each ratio-ensemble member k is paired with an
            # independently sampled subtraction-ensemble member, so all events
            # for ratio member k share the same subtraction network

            # during validation the mean logit across subtraction members is used
            w_sub_tight = sample_subtraction_weights(
                self.num_model,
                tight,
                channels,
                is_data=is_data,
                training=self.training,
            )  # (channels, n_tight)
            w_sub_loose = sample_subtraction_weights(
                self.den_model,
                loose,
                channels,
                is_data=is_data,
                training=self.training,
            )  # (channels, n_loose)

            w_tight_orig = w[tight_mask].unsqueeze(0).expand(channels, -1)  # (channels, n_tight)
            w_loose_orig = w[loose_mask].unsqueeze(0).expand(channels, -1)  # (channels, n_loose)

            w_full = torch.zeros(channels, X.shape[0], device=X.device)
            w_full[:, tight_idx] = w_sub_tight * w_tight_orig
            w_full[:, loose_idx] = w_sub_loose * w_loose_orig

            if self.training and self.poisson_bootstrap:
                w_full = w_full * torch.poisson(torch.ones_like(w_full))

            if self.training and self.weight_clip_quantile is not None:
                clip_vals = torch.quantile(w_full, self.weight_clip_quantile, dim=1, keepdim=True)
                w_full = torch.clamp(w_full, max=clip_vals)

            loss = self.loss_func(y_hat, y_lt, w_full)

        else:
            y_hat = y_hat.flatten()

            tight_reweighted, _, _ = get_num_den_weights(
                self.num_model,
                tight,
                is_data=is_data,
                density_ratio=self.density_ratio,
            )
            w[tight_idx] = tight_reweighted.flatten() * w[tight_mask]

            loose_reweighted, _, _ = get_num_den_weights(
                self.den_model,
                loose,
                is_data=is_data,
                density_ratio=self.density_ratio,
            )
            w[loose_idx] = loose_reweighted.flatten() * w[loose_mask]

            if self.training and self.weight_clip_quantile is not None:
                clip_val = torch.quantile(w, self.weight_clip_quantile)
                w = torch.clamp(w, max=clip_val)

            loss = self.loss_func(y_hat, y_lt, w)

        stage = "train" if self.training else "val"
        batch_size = self.get_batch_size(batch)

        if self.has_ensemble:
            f_mean, f_std = torch_predict_from_ensemble_logits(y_hat)
            r_mean = torch.exp(f_mean)
            r_std = r_mean * f_std
            logits_for_stats = f_mean
            r_for_ess = r_mean
        else:
            logits_for_stats = y_hat
            r_for_ess = torch.exp(y_hat)

        ess = (torch.sum(r_for_ess) ** 2) / (torch.sum(r_for_ess**2) + 1e-9)

        self.log(f"{stage}_ess", ess, batch_size=batch_size)
        self.log(f"{stage}_inv_ess", 1.0 / (ess + 1e-9), batch_size=batch_size)

        self.log(f"{stage}_mean_r", torch.mean(logits_for_stats), batch_size=batch_size)
        self.log(f"{stage}_std_r", torch.std(logits_for_stats), batch_size=batch_size)

        self.log(f"{stage}_max_r", torch.max(logits_for_stats), batch_size=batch_size)
        self.log(f"{stage}_min_r", torch.min(logits_for_stats), batch_size=batch_size)

        if self.has_ensemble:
            self.log(f"{stage}_mean_r_ensemble", torch.mean(r_mean), batch_size=batch_size)
            self.log(f"{stage}_std_r_ensemble", torch.mean(r_std), batch_size=batch_size)

        return loss, y_hat
