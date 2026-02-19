import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Type

import lightning as L
import psutil
import torch
from f9columnar.ml.hdf5_dataloader import FullWeightedBatchType, WeightedBatchType
from f9columnar.utils.helpers import dump_pickle
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig

from seesawml.models.lr_schedulers import select_nn_lr_scheduler
from seesawml.models.optimizers import select_nn_optimizer
from seesawml.models.tracker import Tracker


class BaseLightningModule(ABC, L.LightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        """Abstract base class for pytorch-lightning training.

        Parameters
        ----------
        dataset_conf : dict
            Dataset configuration.
        training_conf : dict
            Training configuration.
        tracker : object or None, optional
            Class for tracking (plots and metrices).

        """
        super().__init__()
        self.dataset_conf, self.model_conf = dataset_conf, model_conf
        self.training_conf = self.model_conf.training_config
        self.run_name = run_name

        self._log_train_memory = self.training_conf.get("log_train_memory", False)

        self.loss_func: Callable[..., torch.Tensor]
        self.model: torch.nn.Module

        self.set_tracker(tracker)

    def set_tracker(self, tracker: Tracker | None) -> None:
        if tracker is None:
            self.tracker = None
        else:
            self.tracker = tracker(self)

    def wrap_model(self, model_wrapper: Type[torch.nn.Module], **kwargs: Any) -> None:
        self.model = model_wrapper(self.model, **kwargs)

    def recompile(self, **kwargs: Any) -> None:
        self.model.compile(**kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = select_nn_optimizer(
            self.training_conf.optimizer.optimizer_name,
            self.parameters(),
            self.training_conf.optimizer.get("optimizer_params", None),
        )

        scheduler_config = self.training_conf.get("scheduler", None)

        if scheduler_config is not None:
            scheduler = select_nn_lr_scheduler(
                scheduler_config.scheduler_name,
                optimizer,
                scheduler_config.get("scheduler_params", None),
            )

            interval = scheduler_config.get("interval", "step")
            monitor = scheduler_config.get("monitor", "val_loss")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": interval,
                },
            }
        else:
            return {"optimizer": optimizer}

    def _training_step(self, loss: torch.Tensor, batch_size: int, reports: dict[str, Any], *args: Any) -> torch.Tensor:
        self.log("train_loss", loss, batch_size=batch_size, prog_bar=True)

        if self.global_step == 0:
            try:
                save_path = os.path.join(
                    self.training_conf["model_save_path"],
                    f"{self.logger._run_name}_reports.p",  # type: ignore[union-attr]
                )
                dump_pickle(save_path, reports)
            except Exception as e:
                logging.warning(f"Could not save reports: {e}")

        if self._log_train_memory:
            process = psutil.Process()
            self.log(
                "train_memory_usage",
                process.memory_info().rss / 1024**2,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        return loss

    @staticmethod
    def _clean_model_str(text: str) -> str:
        pattern = r"\s*\([^)]+\): ParameterList\([\s\S]*?\)\n"

        try:
            text = re.sub(pattern, r"\n", text)
            text = re.sub(r"\n\s*\n", r"\n", text)
        except Exception:
            pass

        return text

    def on_train_start(self) -> None:
        try:
            self.logger.experiment.log_text(self.logger.run_id, self._clean_model_str(str(self)), "model_str.txt")  # type: ignore[union-attr]
        except AttributeError:
            pass

    def on_test_start(self) -> None:
        try:
            self.logger.experiment.log_text(self.logger.run_id, self._clean_model_str(str(self)), "model_str.txt")  # type: ignore[union-attr]
        except AttributeError:
            pass

    def on_train_epoch_end(self) -> None:
        if self.training_conf.get("scheduler", False):
            reduce_lr_on_epoch = self.training_conf["scheduler"].get("reduce_lr_on_epoch", None)
            if reduce_lr_on_epoch is not None:
                self.lr_schedulers().base_lrs = [self.lr_schedulers().base_lrs[0] * reduce_lr_on_epoch]  # type: ignore[union-attr]

        reduce_lr_on_epoch = self.training_conf.get("reduce_lr_on_epoch", None)
        if reduce_lr_on_epoch is not None:
            for param_group in self.optimizers().optimizer.param_groups:  # type: ignore[union-attr]
                param_group["lr"] = param_group["lr"] * reduce_lr_on_epoch

    def on_validation_epoch_end(self) -> None:
        if self.tracker:
            self.tracker.plot(stage="val")

    def on_test_epoch_end(self) -> None:
        if self.tracker:
            self.tracker.plot(stage="test")

    @abstractmethod
    def get_batch_size(self, batch: Any) -> int:
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass


class BaseEventsLightningModule(BaseLightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        super().__init__(dataset_conf, model_conf, run_name, tracker)
        self.save_hyperparameters(ignore=["tracker", "loss_func", "binary_acc", "multi_acc", "model"])

    @abstractmethod
    def get_loss(self, batch: WeightedBatchType, stage: str | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass

    def get_accuracy(self, y_hat: torch.Tensor, batch: WeightedBatchType) -> torch.Tensor | None:
        return None

    def get_batch_size(self, batch: WeightedBatchType) -> int:
        return batch[0].size()[0]

    def training_step(self, batch: WeightedBatchType, *args: Any) -> torch.Tensor:
        batch_size = self.get_batch_size(batch)

        loss, _ = self.get_loss(batch, "train")
        return self._training_step(loss, batch_size, batch[-1], *args)

    def validation_step(self, batch: WeightedBatchType, *args: Any) -> None:
        batch_size = self.get_batch_size(batch)

        loss, y_hat = self.get_loss(batch, "val")
        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)

        if y_hat is not None:
            val_acc = self.get_accuracy(y_hat, batch)
            if val_acc is not None:
                self.log("val_accuracy", val_acc, batch_size=batch_size)

        if self.tracker:
            self.tracker.compute(batch, stage="val")

    def test_step(self, batch: WeightedBatchType, *args: Any) -> None:
        batch_size = self.get_batch_size(batch)

        loss, y_hat = self.get_loss(batch, "test")
        self.log("test_loss", loss, batch_size=batch_size)

        if y_hat is not None:
            test_acc = self.get_accuracy(y_hat, batch)
            if test_acc is not None:
                self.log("test_accuracy", test_acc, batch_size=batch_size)

        if self.tracker:
            self.tracker.compute(batch, stage="test")


class BaseFullLightningModule(BaseLightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        super().__init__(dataset_conf, model_conf, run_name, tracker)
        self.save_hyperparameters(ignore=["tracker", "loss_func", "binary_acc", "multi_acc", "model"])

    @abstractmethod
    def get_loss(
        self, batch: FullWeightedBatchType, stage: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass

    def get_accuracy(self, y_hat: torch.Tensor, batch: FullWeightedBatchType) -> torch.Tensor | None:
        return None

    def get_batch_size(self, batch: FullWeightedBatchType) -> int:
        return batch[0][list(batch[0].keys())[0]][0].size()[0]

    def training_step(self, batch: FullWeightedBatchType, *args: Any) -> torch.Tensor:
        batch_size = self.get_batch_size(batch)

        loss, _ = self.get_loss(batch, "train")
        return self._training_step(loss, batch_size, batch[-1], *args)

    def validation_step(self, batch: FullWeightedBatchType, *args: Any) -> None:
        batch_size = self.get_batch_size(batch)

        loss, y_hat = self.get_loss(batch, "val")
        self.log("val_loss", loss, batch_size=batch_size, prog_bar=True)

        if y_hat is not None:
            val_acc = self.get_accuracy(y_hat, batch)
            if val_acc is not None:
                self.log("val_accuracy", val_acc, batch_size=batch_size)

        if self.tracker:
            self.tracker.compute(batch, stage="val")

    def test_step(self, batch: FullWeightedBatchType, *args: Any) -> None:
        batch_size = self.get_batch_size(batch)

        loss, y_hat = self.get_loss(batch, "test")
        self.log("test_loss", loss, batch_size=batch_size)

        if y_hat is not None:
            test_acc = self.get_accuracy(y_hat, batch)
            if test_acc is not None:
                self.log("test_accuracy", test_acc, batch_size=batch_size)

        if self.tracker:
            self.tracker.compute(batch, stage="test")


class BaseBatchCompletedLightningModule(BaseEventsLightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        tracker: Tracker | None = None,
    ) -> None:
        super().__init__(dataset_conf, model_conf, run_name, tracker)

    @abstractmethod
    def get_loss(self, batch: WeightedBatchType, stage: str | None = None) -> Any:
        pass

    @staticmethod
    def _cat_batch(X, X_cat, y, y_cat, w, w_cat, y_lt, y_lt_cat):
        X_cat, y_cat = torch.cat((X_cat, X), dim=0), torch.cat((y_cat, y), dim=0)
        w_cat, y_lt_cat = torch.cat((w_cat, w), dim=0), torch.cat((y_lt_cat, y_lt), dim=0)

        return X_cat, y_cat, w_cat, y_lt_cat

    @staticmethod
    def _cut_batch(X_cat, y_cat, w_cat, y_lt_cat, cut):
        X_cat, y_cat = X_cat[:cut], y_cat[:cut]
        w_cat, y_lt_cat = w_cat[:cut], y_lt_cat[:cut]

        return X_cat, y_cat, w_cat, y_lt_cat

    def prepare_cat_batch(
        self, dataloader_iter: Iterator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]] | None:
        batch: WeightedBatchType

        current_batch_size = 0
        while True:
            try:
                batch, _, _ = next(dataloader_iter)
            except StopIteration:
                return None

            X, y, w, y_lt, reports = batch
            actual_batch_size, expected_batch_size = len(X), reports["batch_size"]

            if current_batch_size == 0:
                X_cat, y_cat, w_cat, y_lt_cat = X, y, w, y_lt

                if actual_batch_size == expected_batch_size:
                    break
            else:
                X_cat, y_cat, w_cat, y_lt_cat = self._cat_batch(X, X_cat, y, y_cat, w, w_cat, y_lt, y_lt_cat)

            current_batch_size += actual_batch_size

            if current_batch_size == expected_batch_size:
                break

            if current_batch_size > expected_batch_size:
                X_cat, y_cat, w_cat, y_lt_cat = self._cut_batch(X_cat, y_cat, w_cat, y_lt_cat, expected_batch_size)
                break

        X_cat, y_cat = X_cat.to(self.device), y_cat.to(self.device)
        w_cat, y_lt_cat = w_cat.to(self.device), y_lt_cat.to(self.device)

        cat_batch = (X_cat, y_cat, w_cat, y_lt_cat, reports)

        return cat_batch

    def training_step(self, dataloader_iter: Iterator) -> torch.Tensor | None:  # type: ignore[override]
        cat_batch = self.prepare_cat_batch(dataloader_iter)
        if cat_batch is None:
            return None

        loss, extra = self.get_loss(cat_batch, "train")
        self.log("train_loss", loss, batch_size=cat_batch[0].size()[0])

        if type(extra) is dict:
            for key, value in extra.items():
                self.log(f"train_{key}", value, batch_size=cat_batch[0].size()[0])

        if self.global_step == 0:
            try:
                save_path = os.path.join(
                    self.training_conf["model_save_path"],
                    f"{self.logger._run_name}_reports.p",  # type: ignore[union-attr]
                )
                dump_pickle(save_path, cat_batch[-1])
            except Exception as e:
                logging.warning(f"Could not save reports: {e}")

        return loss

    def validation_step(self, dataloader_iter: Iterator) -> torch.Tensor | None:  # type: ignore[override]
        cat_batch = self.prepare_cat_batch(dataloader_iter)
        if cat_batch is None:
            return None

        loss, _ = self.get_loss(cat_batch, "val")
        self.log("val_loss", loss, batch_size=cat_batch[0].size()[0])

        if self.tracker:
            self.tracker.compute(cat_batch, stage="val")

        return loss
