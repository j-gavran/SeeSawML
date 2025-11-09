import hashlib
import logging
import os
from typing import Any

import torch
from f9columnar.ml.dataloader_helpers import ColumnSelection
from f9columnar.ml.hdf5_dataloader import FullWeightedBatchType, WeightedBatchType
from f9columnar.utils.helpers import load_pickle
from omegaconf import DictConfig
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

from seesaw.models.disco import distance_corr
from seesaw.models.losses import select_nn_loss
from seesaw.models.nn_modules import BaseEventsLightningModule, BaseFullLightningModule
from seesaw.models.tracker import Tracker
from seesaw.models.utils import build_network


class BaseSigBkgNNClassifier:
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        run_name: str | None = None,
        use_mc_weights: bool = False,
        use_class_weights: bool = False,
    ) -> None:
        self.dataset_conf = dataset_conf
        self.model_conf = model_conf
        self.model, self.model_name, selection = build_network(dataset_conf, model_conf, run_name=run_name)

        if type(model_conf.training_config.loss) is str:
            loss_name = model_conf.training_config.loss
            loss_params = {}
        else:
            loss_name = model_conf.training_config.loss.loss_name
            loss_params = dict(model_conf.training_config.loss.get("loss_params", {}))

        logging.info(f"Using {loss_name} loss function with parameters: {loss_params}.")
        self.loss_func = select_nn_loss(loss_name, **loss_params)

        classes = dataset_conf.get("classes", None)
        if classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = 2

        self.binary_acc: BinaryAccuracy | None = None
        self.multi_acc: MulticlassAccuracy | None = None

        if self.num_classes == 2:
            self.binary_acc = BinaryAccuracy()
        else:
            self.multi_acc = MulticlassAccuracy(num_classes=self.num_classes)

        classes = dataset_conf.get("classes", None)
        if classes is not None:
            self.is_multiclass = True if len(classes) > 2 else False
        else:
            self.is_multiclass = False

        self.setup_weights(use_mc_weights, use_class_weights)
        self.setup_disco(selection)

    def setup_weights(self, use_mc_weights: bool = False, use_class_weights: bool = False) -> None:
        self.use_mc_weights = use_mc_weights
        self.use_class_weights = use_class_weights

        if use_class_weights:
            classes = self.dataset_conf.get("classes", None)
            if classes is None:
                raise ValueError("No classes defined in the dataset configuration for class weights.")

            class_weights_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "class_weights")

            hash_name = "".join(str(classes)) + str(self.dataset_conf.files)
            class_weights_file_name = hashlib.md5(hash_name.encode()).hexdigest()

            logging.info("Loading class weights.")
            try:
                self.class_weights = load_pickle(os.path.join(class_weights_dir, f"{class_weights_file_name}.p"))
            except FileNotFoundError:
                logging.error(
                    f"Class weights file not found in {class_weights_dir}. Please run calculate_class_weights first."
                )
                raise FileNotFoundError("Could not load class weights.")

    def setup_disco(self, selection: ColumnSelection) -> None:
        disco_config = self.model_conf.training_config.get("disco", None)

        if disco_config is None:
            self.use_disco = False
            return None

        used_columns = selection["events"].offset_used_columns
        disco_variables = disco_config["variables"]

        disco_idx = []
        for v in disco_variables:
            disco_idx.append(used_columns.index(v))

        if len(disco_idx) != len(disco_variables):
            raise ValueError(f"Disco variables {disco_variables} not found in used columns {used_columns}. ")

        self.disco_idx = torch.tensor(disco_idx, dtype=torch.long)
        self.disco_lambda = disco_config["lambda"]
        self.disco_power = disco_config.get("power", 1.0)
        self.disco_weighted = disco_config.get("weighted", False)

        logging.info(f"Decorrelating variables: {disco_config['variables']} using disco.")

        if self.is_multiclass:
            self.disco_multiclass_reduction = disco_config.get("multiclass_reduction", "logits")
            logging.info(f"Using multiclass disco reduction method: {self.disco_multiclass_reduction}.")
            if self.disco_weighted:
                self.disco_weighted = False
                logging.warning("Setting disco_weighted to False for multiclass disco!")

        self.dcorr = 0.0
        self.use_disco = True

    def get_dcorr(
        self,
        X: torch.Tensor,
        y_hat: torch.Tensor,
        y_classes: torch.Tensor,
        reports: dict[str, Any],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signal_idx = reports["class_labels"]["signal"]

        y_bkg_idx = torch.where(y_classes != signal_idx)[0]
        X, y_hat = X[y_bkg_idx], y_hat[y_bkg_idx]

        if weights is not None:
            weights = weights[y_bkg_idx]

        disco_vars = X[:, self.disco_idx]

        dcorrs = []

        if not self.is_multiclass:
            for i in range(len(self.disco_idx)):
                target_var = disco_vars[:, i].squeeze()
                dcorrs.append(distance_corr(target_var, y_hat, self.disco_power, weights))

            dcorr = torch.stack(dcorrs).sum()

            return dcorr

        elif self.disco_multiclass_reduction == "entropy":
            probs = torch.softmax(y_hat, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)

            for i in range(disco_vars.shape[1]):
                target_var = disco_vars[:, i].squeeze()
                dcorrs.append(distance_corr(target_var, entropy, self.disco_power, weights))

            dcorr = torch.stack(dcorrs).sum()

            return dcorr

        elif self.disco_multiclass_reduction == "logits":
            for i in range(disco_vars.shape[1]):
                target_var = disco_vars[:, i].squeeze()

                logit_dcorrs = []
                for c in range(y_hat.shape[1]):
                    dcorr = distance_corr(target_var, y_hat[:, c], self.disco_power, weights)
                    logit_dcorrs.append(dcorr)

                logit_dcorr = torch.stack(logit_dcorrs).sum()
                dcorrs.append(logit_dcorr)

            dcorr = torch.stack(dcorrs).sum()

            return dcorr

        else:
            raise ValueError(f"Invalid disco multiclass reduction method: {self.disco_multiclass_reduction}. ")

    def get_classifier_loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        y_hat: torch.Tensor,
        y_classes: torch.Tensor,
        reports: dict[str, Any],
    ) -> torch.Tensor:
        if not self.is_multiclass:
            y_hat = y_hat.squeeze(1)

        loss = self.loss_func(y_hat, y)

        if self.use_mc_weights:
            loss = torch.abs(w) * loss

        if self.use_class_weights:
            class_weights = torch.zeros_like(y_classes, dtype=X.dtype)

            for class_label, class_weight in self.class_weights.items():
                class_weights[y_classes == class_label] = class_weight

            loss = class_weights * loss

        if self.use_disco:
            disco_weights = torch.ones_like(w, dtype=X.dtype)

            if self.disco_weighted and self.use_mc_weights:
                disco_weights = disco_weights * torch.abs(w)
            if self.disco_weighted and self.use_class_weights:
                disco_weights = disco_weights * class_weights

            dcorr = self.disco_lambda * self.get_dcorr(X, y_hat, y_classes, reports, weights=disco_weights)
            loss = torch.mean(loss) + dcorr
            self.dcorr = dcorr.item()
        else:
            loss = torch.mean(loss)

        return loss

    def on_validation_epoch_end(self):
        if self.tracker:
            self.tracker.plot(stage="val")


class SigBkgEventsNNClassifier(BaseSigBkgNNClassifier, BaseEventsLightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        tracker: Tracker | None = None,
        run_name: str | None = None,
        use_mc_weights: bool = False,
        use_class_weights: bool = False,
    ) -> None:
        BaseEventsLightningModule.__init__(
            self,
            dataset_conf,
            model_conf,
            tracker=tracker,
        )
        BaseSigBkgNNClassifier.__init__(
            self,
            dataset_conf,
            model_conf,
            run_name=run_name,
            use_mc_weights=use_mc_weights,
            use_class_weights=use_class_weights,
        )

    def forward(self, X: torch.Tensor, use_sigmoid: bool = False, use_softmax: bool = False) -> torch.Tensor:
        if self.training:
            return self.model(X)

        if use_sigmoid:
            return torch.sigmoid(self.model(X))

        if use_softmax:
            return torch.softmax(self.model(X), dim=1)

        return self.model(X)

    def get_loss(self, batch: WeightedBatchType, stage: str | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        X, y, w, y_classes, reports = batch

        y_hat = self(X)

        loss = self.get_classifier_loss(X, y, w, y_hat, y_classes, reports)

        if self.use_disco:
            self.log(f"{stage}_dcorr", self.dcorr, batch_size=self.get_batch_size(batch))

        return loss, y_hat

    def get_accuracy(self, y_hat: torch.Tensor, batch: WeightedBatchType) -> torch.Tensor:
        if self.binary_acc is not None:
            return self.binary_acc(y_hat.squeeze(1), batch[1])
        elif self.multi_acc is not None:
            return self.multi_acc(y_hat, torch.argmax(batch[1], dim=1))
        else:
            raise RuntimeError("No accuracy metric defined for this model!")


class SigBkgFullNNClassifier(BaseSigBkgNNClassifier, BaseFullLightningModule):
    def __init__(
        self,
        dataset_conf: DictConfig,
        model_conf: DictConfig,
        tracker: Tracker | None = None,
        run_name: str | None = None,
        use_mc_weights: bool = False,
        use_class_weights: bool = False,
    ) -> None:
        BaseFullLightningModule.__init__(
            self,
            dataset_conf,
            model_conf,
            tracker=tracker,
        )
        BaseSigBkgNNClassifier.__init__(
            self,
            dataset_conf,
            model_conf,
            run_name=run_name,
            use_mc_weights=use_mc_weights,
            use_class_weights=use_class_weights,
        )

    def forward(
        self,
        X_events: torch.Tensor,
        Xs: list[torch.Tensor],
        use_sigmoid: bool = False,
        use_softmax: bool = False,
    ) -> torch.Tensor:
        if self.training:
            return self.model(X_events, *Xs)

        if use_sigmoid:
            return torch.sigmoid(self.model(X_events, *Xs))

        if use_softmax:
            return torch.softmax(self.model(X_events, *Xs), dim=1)

        return self.model(X_events, *Xs)

    def get_loss(
        self, batch: FullWeightedBatchType, stage: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        Xs: list[torch.Tensor] = []

        for k in batch[0].keys():
            if k != "events":
                Xs.append(batch[0][k][0])

        X_events = batch[0]["events"][0]
        y, w, y_classes = batch[0]["events"][1], batch[0]["events"][2], batch[0]["events"][3]

        if y is None or w is None or y_classes is None:
            raise ValueError("y, w, or y_classes is None, cannot compute loss!")

        y_hat = self(X_events, Xs)

        loss = self.get_classifier_loss(X_events, y, w, y_hat, y_classes, batch[1])

        if self.use_disco:
            self.log(f"{stage}_dcorr", self.dcorr, batch_size=self.get_batch_size(batch))

        return loss, y_hat

    def get_accuracy(self, y_hat: torch.Tensor, batch: FullWeightedBatchType) -> torch.Tensor:
        y_true = batch[0]["events"][1]

        if y_true is None:
            raise ValueError("y_true is None, cannot compute accuracy!")

        if self.binary_acc is not None:
            return self.binary_acc(y_hat.squeeze(1), y_true)
        elif self.multi_acc is not None:
            return self.multi_acc(y_hat, torch.argmax(y_true, dim=1))
        else:
            raise RuntimeError("No accuracy metric defined for this model!")
