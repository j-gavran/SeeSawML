from abc import abstractmethod
from typing import Any

import torch
from f9columnar.ml.hdf5_dataloader import WeightedBatchType
from omegaconf import DictConfig
from torchmetrics.classification import BinaryAccuracy

from seesaw.fakes.models.crack_veto_model import get_crack_veto_model
from seesaw.fakes.models.loss import DensityRatio, DensityRatioLoss
from seesaw.fakes.utils import get_num_den_weights
from seesaw.models.nn_modules import BaseBatchCompletedLightningModule
from seesaw.models.tracker import Tracker
from seesaw.models.utils import build_network


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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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

        self.save_hyperparameters(ignore=["tracker", "loss_func", "model"])

    def get_accuracy(self, y_hat: torch.Tensor, batch: WeightedBatchType) -> torch.Tensor:
        return self.binary_acc(y_hat, batch[1])

    def get_loss(self, batch: WeightedBatchType, stage: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        X, y, w, _, reports = batch

        if reports["use_data"] is False or reports["use_mc"] is False:
            raise RuntimeError("Invalid data/mc configuration!")

        y_hat = self(X).flatten()

        loss = self.loss_func(y_hat, y, w)

        if self.training:
            stage = "train"
        else:
            stage = "val"

        ess = (torch.sum(y_hat) ** 2) / (torch.sum(y_hat**2) + 1e-9)

        self.log(f"{stage}_ess", ess, batch_size=self.get_batch_size(batch))
        self.log(f"{stage}_inv_ess", 1.0 / (ess + 1e-9), batch_size=self.get_batch_size(batch))

        self.log(f"{stage}_mean_r", torch.mean(y_hat), batch_size=self.get_batch_size(batch))
        self.log(f"{stage}_std_r", torch.std(y_hat), batch_size=self.get_batch_size(batch))

        self.log(f"{stage}_max_r", torch.max(y_hat), batch_size=self.get_batch_size(batch))
        self.log(f"{stage}_min_r", torch.min(y_hat), batch_size=self.get_batch_size(batch))

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

        if num_model is not None:
            self.num_model = num_model.eval()

        if den_model is not None:
            self.den_model = den_model.eval()

        self.save_hyperparameters(ignore=["tracker", "loss_func", "density_ratio", "model", "num_model", "den_model"])

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

        # tight -> numerator
        tight_mask = y_lt == 1.0
        tight_idx = torch.argwhere(tight_mask).flatten()

        tight = X[tight_mask]

        tight_reweighted = get_num_den_weights(
            self.num_model,
            tight,
            is_data=is_data,
            density_ratio=self.density_ratio,
        )
        tight_reweighted = tight_reweighted.flatten()  # type: ignore
        tight_w = tight_reweighted * w[tight_mask]

        w[tight_idx] = tight_w

        # loose -> denominator
        loose_mask = y_lt == 0.0
        loose_idx = torch.argwhere(loose_mask).flatten()

        loose = X[loose_mask]
        loose_reweighted = get_num_den_weights(
            self.den_model,
            loose,
            is_data=is_data,
            density_ratio=self.density_ratio,
        )
        loose_reweighted = loose_reweighted.flatten()  # type: ignore
        loose_w = loose_reweighted * w[loose_mask]

        w[loose_idx] = loose_w

        y_hat = self(X).flatten()

        loss = self.loss_func(y_hat, y_lt, w)

        if self.training:
            stage = "train"
        else:
            stage = "val"

        ess = (torch.sum(y_hat) ** 2) / (torch.sum(y_hat**2) + 1e-9)

        self.log(f"{stage}_ess", ess, batch_size=self.get_batch_size(batch))
        self.log(f"{stage}_inv_ess", 1.0 / (ess + 1e-9), batch_size=self.get_batch_size(batch))

        self.log(f"{stage}_mean_r", torch.mean(y_hat), batch_size=self.get_batch_size(batch))
        self.log(f"{stage}_std_r", torch.std(y_hat), batch_size=self.get_batch_size(batch))

        self.log(f"{stage}_max_r", torch.max(y_hat), batch_size=self.get_batch_size(batch))
        self.log(f"{stage}_min_r", torch.min(y_hat), batch_size=self.get_batch_size(batch))

        return loss, y_hat
