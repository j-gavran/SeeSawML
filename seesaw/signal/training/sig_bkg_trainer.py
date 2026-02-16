import logging
import os
from typing import Any

import hydra
import lightning as L
from f9columnar.ml.dataloader_helpers import get_hdf5_columns, get_hdf5_metadata
from f9columnar.ml.hdf5_dataloader import events_collate_fn, full_collate_fn
from f9columnar.ml.lightning_data_module import LightningHdf5DataModule
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

from seesaw.models.calibration import get_calibration_split, get_calibration_wrapper
from seesaw.utils.constants import add_type_fields_to_features
from seesaw.models.nn_modules import BaseLightningModule
from seesaw.models.tracker import Tracker
from seesaw.models.utils import load_model_from_config
from seesaw.signal.models.sig_bkg_classifiers import SigBkgEventsNNClassifier, SigBkgFullNNClassifier
from seesaw.signal.training.trackers import (
    JaggedSigBkgClassifierTracker,
    JaggedSigBkgMulticlassClassifierTracker,
    SigBkgClassifierTracker,
    SigBkgMulticlassClassifierTracker,
)
from seesaw.signal.utils import get_classifier_labels, handle_events_signal_dataset, handle_full_signal_dataset
from seesaw.utils.helpers import setup_analysis_dirs, verify_num_workers
from seesaw.utils.loggers import log_hydra_config, setup_logger
from seesaw.utils.trainer_utils import (
    get_callbacks,
    get_logger,
    get_T_0_from_scheduler_config,
    get_trainer,
    set_run_name,
)


def get_signal_data_module(
    dataset_conf: DictConfig,
    dataset_name: str,
    model_name: str,
    events_only: bool = True,
    is_calibration: bool = False,
    model_conf: DictConfig | None = None,
) -> L.LightningDataModule:
    # Auto-add type fields if valid_type_values is configured
    features = list(dataset_conf.features)
    original_count = len(features)
    if model_conf is not None:
        valid_type_values = model_conf.architecture_config.get("valid_type_values", None)
        features = add_type_fields_to_features(features, valid_type_values)
        if len(features) > original_count:
            logging.info(f"[green]DataModule features expanded: {original_count} -> {len(features)} features[/green]")

    feature_scaling_config = dataset_conf.get("feature_scaling", None)

    if feature_scaling_config is not None:
        feature_scaling_kwargs = {
            "numer_scaler_type": feature_scaling_config.get("numer_scaler_type", None),
            "categ_scaler_type": feature_scaling_config.get("categ_scaler_type", None),
            "scaler_path": feature_scaling_config.get("save_path", None),
            "scalers_extra_hash": str(dataset_conf.files),
        }
    else:
        logging.warning("Feature scaling is disabled!")
        feature_scaling_kwargs = {}

    dataset_kwargs = dict(dataset_conf["dataset_kwargs"]) | feature_scaling_kwargs

    if dataset_kwargs.get("imbalanced_sampler", None) is not None:
        if dataset_conf.get("use_class_weights", False):
            raise ValueError("Imbalanced sampling cannot be used with class weights!")

        if dataset_conf.get("use_mc_weights", False):
            raise ValueError("Imbalanced sampling cannot be used with MC weights!")

    if dataset_kwargs.get("imbalanced_sampler_kwargs", None) is not None:
        imbalanced_sampler_kwargs = OmegaConf.to_container(dataset_kwargs["imbalanced_sampler_kwargs"], resolve=True)
        dataset_kwargs["imbalanced_sampler_kwargs"] = imbalanced_sampler_kwargs

    metadata = get_hdf5_metadata(dataset_conf.files, resolve_path=True)
    labels = metadata.get("labels", None)

    if labels is not None:
        classes = dataset_conf.get("classes", None)
        if classes is not None:
            class_labels, remap_labels, _ = get_classifier_labels(classes, labels)
            dataset_kwargs["remap_labels"] = remap_labels
            dataset_kwargs["max_label"] = max(labels.values())
            dataset_kwargs["class_labels"] = class_labels

    if events_only:
        logging.info("[green]Using events-only (flat) dataset configuration.")
        dataset_kwargs["setup_func"] = handle_events_signal_dataset
        collate_func = events_collate_fn
    else:
        logging.info("[green]Using full (flat + jagged) dataset configuration.")
        dataset_kwargs["setup_func"] = handle_full_signal_dataset
        collate_func = full_collate_fn

    dataset_kwargs["events_only"] = events_only

    dataloader_kwargs = dict(dataset_conf.dataloader_kwargs)

    dm_name = f"{dataset_name} - {model_name[0].upper() + model_name[1:]}"

    stage_split_piles = dataset_conf.stage_split_piles
    if "calib" in stage_split_piles:
        logging.info("[yellow]Calibration split detected in stage_split_piles.")
        stage_split_piles = get_calibration_split(dict(stage_split_piles), add_calib_train=is_calibration)

    verify_num_workers(dataloader_kwargs.get("num_workers", -1), stage_split_piles)

    dm_iter = LightningHdf5DataModule(
        dm_name,
        dataset_conf.files,
        features,
        stage_split_piles=stage_split_piles,
        shuffle=True,
        collate_fn=collate_func,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    return dm_iter


def build_sig_bkg_model(
    model_conf: DictConfig,
    dataset_conf: DictConfig,
    run_name: str,
    tracker: Tracker | None = None,
    events_only: bool = True,
) -> BaseLightningModule:
    logging.info(f"Building {model_conf.name}.")

    if events_only:
        return SigBkgEventsNNClassifier(
            dataset_conf=dataset_conf,
            model_conf=model_conf,
            tracker=tracker,
            run_name=run_name,
            use_mc_weights=dataset_conf.get("use_mc_weights", False),
            use_class_weights=dataset_conf.get("use_class_weights", False),
        )
    else:
        return SigBkgFullNNClassifier(
            dataset_conf=dataset_conf,
            model_conf=model_conf,
            tracker=tracker,
            run_name=run_name,
            use_mc_weights=dataset_conf.get("use_mc_weights", False),
            use_class_weights=dataset_conf.get("use_class_weights", False),
        )


def load_sig_bkg_model(
    config: DictConfig,
    events_only: bool = True,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> tuple[BaseLightningModule, str]:
    if "_calib.ckpt" in config.model_config.load_checkpoint:
        calib_wrapper = get_calibration_wrapper(config.model_config.calibration_config.method, events_only)
    else:
        calib_wrapper = None

    if events_only:
        return load_model_from_config(
            config,
            SigBkgEventsNNClassifier,
            checkpoint_path=checkpoint_path,
            model_wrapper=calib_wrapper,
            **kwargs,
        )
    else:
        return load_model_from_config(
            config,
            SigBkgFullNNClassifier,
            checkpoint_path=checkpoint_path,
            model_wrapper=calib_wrapper,
            **kwargs,
        )


def verify_loss(is_multiclass: bool, model_conf: DictConfig) -> None:
    loss_conf = model_conf.training_config.loss

    if type(loss_conf) is str:
        loss_name = loss_conf
    else:
        loss_name = loss_conf.loss_name

    if is_multiclass:
        if loss_name not in ["ce", "CrossEntropyLoss", "multiclass_focal", "MulticlassFocalLoss", "mse", "MSELoss"]:
            raise ValueError(f"Loss function {loss_name} is not supported for multiclass classification!")
    else:
        if loss_name not in ["bce", "BCEWithLogitsLoss", "sigmoid_focal", "SigmoidFocalLoss", "mse", "MSELoss"]:
            logging.warning(f"Loss function {loss_name} is not *officially* supported for binary classification!")


def add_flat_model_config(config: DictConfig) -> None:
    if "flat_model_config" not in config.model_config:
        return None

    logging.info("Adding flat model configuration to the main model configuration.")
    with open_dict(config):
        flat_model_config = config.model_config.pop("flat_model_config")
        config.model_config.architecture_config.flat_model_config = flat_model_config


def sig_bkg_trainer(config: DictConfig) -> None:
    setup_analysis_dirs(config)
    log_hydra_config(config)

    add_flat_model_config(config)

    experiment_config = config.experiment_config
    experiment_config = set_run_name(experiment_config, config.model_config)

    dataset_config, model_config = config.dataset_config, config.model_config

    dataset_name = dataset_config.name

    model_name = model_config.name
    if model_name != "sigBkgClassifier":
        raise NotImplementedError(f"Model {model_name} not implemented!")

    columns_dct = get_hdf5_columns(dataset_config.files, resolve_path=True)
    events_only = not any(
        c in dataset_config.features and dataset_name != "events"
        for dataset_name, columns in columns_dct.items()
        for c in columns
    )

    train_stage = True

    load_checkpoint = model_config.load_checkpoint

    if load_checkpoint is not None:
        model_save_file = os.path.join(model_config.training_config.model_save_path, load_checkpoint)

        if not model_save_file.endswith(".ckpt"):
            raise ValueError(f"Checkpoint {load_checkpoint} does not have a valid .ckpt extension!")

        if not os.path.exists(model_save_file):
            logging.warning(f"Checkpoint {load_checkpoint} does not exist! Training from scratch.")
            load_checkpoint = None
        else:
            logging.info(f"Loading from checkpoint {load_checkpoint}.")

    if load_checkpoint is not None:
        try:
            model, _ = load_sig_bkg_model(config, events_only=events_only)
            logging.info("Sig/bkg model is ready to be used.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")

    continue_train = model_config.get("continue_training", False)

    if load_checkpoint is None and continue_train:
        raise ValueError("Cannot continue training without a valid checkpoint.")

    if load_checkpoint is not None and not continue_train:
        train_stage = False
        logging.info("Proceeding to testing stage.")

    classes = dataset_config.get("classes", None)
    if classes is not None and len(classes) > 2:
        logging.info("[yellow]Detected multiclass classification problem.")
        is_multiclass = True
    else:
        logging.info("[yellow]Detected binary classification problem.")
        is_multiclass = False

    tracker: Tracker | None
    significance_conf = config.get("significance_config", None)

    if is_multiclass:
        if events_only:
            tracker = SigBkgMulticlassClassifierTracker(
                experiment_config,
                dataset_config,
                model_config,
                config.plotting_config,
                tracker_path=os.path.join(experiment_config.tracker_path, model_name),
                significance_conf=significance_conf,
            )
        else:
            tracker = JaggedSigBkgMulticlassClassifierTracker(
                experiment_config,
                dataset_config,
                model_config,
                config.plotting_config,
                tracker_path=os.path.join(experiment_config.tracker_path, model_name),
                significance_conf=significance_conf,
            )
    else:
        if events_only:
            tracker = SigBkgClassifierTracker(
                experiment_config,
                dataset_config,
                model_config,
                config.plotting_config,
                tracker_path=os.path.join(experiment_config.tracker_path, model_name),
            )
        else:
            tracker = JaggedSigBkgClassifierTracker(
                experiment_config,
                dataset_config,
                model_config,
                config.plotting_config,
                tracker_path=os.path.join(experiment_config.tracker_path, model_name),
            )

    verify_loss(is_multiclass, model_config)

    if train_stage and not continue_train:
        model = build_sig_bkg_model(
            model_config,
            dataset_config,
            run_name=experiment_config.run_name,
            tracker=tracker,
            events_only=events_only,
        )

    experiment_name = experiment_config.experiment_name
    if train_stage:
        logging.info(f"[bold green]Starting experiment: {experiment_name}.")
    else:
        experiment_name = f"{experiment_name}_test"

    callbacks = get_callbacks(
        model_config.training_config,
        experiment_config.run_name,
        monitor=model_config.training_config.monitor,
        monitor_mode=model_config.training_config.get("monitor_mode", "min"),
        tqdm_refresh_rate=experiment_config.get("tqdm_refresh_rate", 100),
    )

    logger = get_logger(
        experiment_name,
        experiment_config.run_name,
        experiment_config.save_dir,
        comet_api_key=experiment_config.get("comet_api_key", None),
        comet_project_name=experiment_config.get("comet_project_name", None),
    )

    T_0 = get_T_0_from_scheduler_config(model_config)
    trainer = get_trainer(experiment_config, model_config.training_config, logger, callbacks, val_check_interval=T_0)

    dm = get_signal_data_module(
        dataset_config, dataset_name, model_name, events_only=events_only, model_conf=model_config
    )

    if continue_train:
        logging.info("[yellow]Continuing training from the loaded model...")

    if train_stage:
        trainer.fit(model, dm)

        best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
        logging.info(f"Best model saved at: {best_model_path}")

        logging.info("[green]Done training!")
        return None

    if dataset_config.stage_split_piles.get("test", 0) > 0:
        logging.info("[bold green]Testing trained model.")
        model.set_tracker(tracker)
        trainer.test(model, dm)
    else:
        logging.info("No test data available. Skipping testing.")

    logging.info("[green]Done testing!")


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)

    seed_everything(config.experiment_config.seed, workers=True)

    sig_bkg_trainer(config)


if __name__ == "__main__":
    main()
