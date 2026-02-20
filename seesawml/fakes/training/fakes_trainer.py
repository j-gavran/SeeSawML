import logging
import os
from typing import Any

import hydra
import lightning as L
import optuna
import torch
from f9columnar.ml.hdf5_dataloader import events_collate_fn, get_column_selection
from f9columnar.ml.lightning_data_module import LightningHdf5DataModule
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig
from optuna.integration import PyTorchLightningPruningCallback

from seesawml.fakes.models.dre_classifiers import NumDenClassifier, RatioClassifier
from seesawml.fakes.models.pt_sliced_model import get_numer_scaler, get_pt_idx, load_pt_sliced_model, scale_pt_slice
from seesawml.fakes.training.trackers import NumDenTracker, RatioTracker
from seesawml.fakes.utils import handle_fakes_dataset
from seesawml.models.nn_modules import BaseLightningModule
from seesawml.models.tracker import Tracker
from seesawml.models.utils import load_model_from_config
from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.loggers import log_hydra_config, setup_logger
from seesawml.utils.trainer_utils import (
    get_callbacks,
    get_logger,
    get_T_0_from_scheduler_config,
    get_trainer,
    set_run_name,
)
from seesawml.utils.tuning import suggest_to_config


def get_fakes_data_module(
    dataset_conf: DictConfig, dataset_name: str, model_name: str, config: DictConfig | None = None
) -> LightningHdf5DataModule:
    dataset_kwargs: dict[str, Any]

    if model_name == "numModel":
        dataset_kwargs = {
            "use_data": True,
            "use_mc": True,
            "use_loose": False,
            "use_tight": True,
        }
    elif model_name == "denModel":
        dataset_kwargs = {
            "use_data": True,
            "use_mc": True,
            "use_loose": True,
            "use_tight": False,
        }
    elif model_name == "ratioModel":
        use_data_in_ratio = dataset_conf.dataset_kwargs.use_data_in_ratio
        dataset_kwargs = {
            "use_data": True if use_data_in_ratio else False,
            "use_mc": False if use_data_in_ratio else True,
            "use_loose": True,
            "use_tight": True,
        }
    else:
        raise ValueError(f"Model {model_name} is not supported for dataset {dataset_name}!")

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

    dataset_kwargs = dataset_kwargs | feature_scaling_kwargs | dict(dataset_conf["dataset_kwargs"])
    dataset_kwargs["setup_func"] = handle_fakes_dataset

    pt_cut = dataset_conf.dataset_kwargs.get("pt_cut", None)
    if pt_cut is not None and pt_cut["min"] is not None and pt_cut["max"] is not None:
        if config is None:
            raise ValueError("Config must be provided for pt sliced model!")

        events_column = get_column_selection(dataset_conf.files, dataset_conf.features)["events"]

        unsorted_column_names = set(events_column.used_columns) - set(events_column.extra_columns)
        column_names = [str(c) for c in events_column.used_columns if c in unsorted_column_names]

        dataset_kwargs["pt_idx"] = get_pt_idx(column_names)

        numer_column_names = events_column.numer_columns
        numer_column_idx = events_column.offset_numer_columns_idx

        pt_numer_idx = get_pt_idx(numer_column_names)
        numer_scaler = get_numer_scaler(config, numer_column_names, extra_hash=config.dataset_config.files)

        pt_slice_float = (float(pt_cut["min"]), float(pt_cut["max"]))

        if numer_scaler is not None:
            pt_slice_scaled = scale_pt_slice(numer_scaler, pt_slice_float, pt_numer_idx, numer_column_idx)
        else:
            pt_slice_scaled = pt_slice_float

        logging.info(f"Using pt slice: {pt_slice_float} GeV, scaled: {pt_slice_scaled}.")
        dataset_kwargs["pt_cut"] = pt_slice_scaled

    dataloader_kwargs = dict(dataset_conf.dataloader_kwargs)

    dm_name = f"{dataset_name}{model_name[0].upper() + model_name[1:]}DataModule"
    logging.info(f"Setting up {dm_name}.")

    dm = LightningHdf5DataModule(
        dm_name,
        dataset_conf.files,
        dataset_conf.features,
        stage_split_piles=dataset_conf.stage_split_piles,
        shuffle=True,
        collate_fn=events_collate_fn,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    return dm


def build_num_den_model(
    model_conf: DictConfig,
    dataset_conf: DictConfig,
    tracker: Tracker | None = None,
    run_name: str | None = None,
) -> NumDenClassifier:
    logging.info(f"Building {model_conf.name}.")

    return NumDenClassifier(
        dataset_conf,
        model_conf,
        run_name=run_name,
        tracker=tracker,
    )


def load_num_den_model(conf: DictConfig) -> tuple[BaseLightningModule, str]:
    if conf.model_config.get("load_checkpoint", None) is not None:
        return load_model_from_config(conf, NumDenClassifier)
    else:
        raise ValueError("No checkpoint provided for num/den model!")


def build_ratio_model(
    config: DictConfig, tracker: Tracker | None = None, run_name: str | None = None
) -> RatioClassifier:
    model_conf, dataset_conf = config.model_config, config.dataset_config

    num_conf, den_conf = model_conf.num_config, model_conf.den_config

    logging.info(f"Building {model_conf.name}.")

    num_model, _ = load_model_from_config(config, NumDenClassifier, model_config=num_conf)
    den_model, _ = load_model_from_config(config, NumDenClassifier, model_config=den_conf)

    return RatioClassifier(
        dataset_conf,
        model_conf,
        run_name=run_name,
        num_model=num_model,
        den_model=den_model,
        tracker=tracker,
    )


def load_ratio_model(conf: DictConfig) -> tuple[BaseLightningModule | torch.nn.Module, str]:
    if conf.model_config.get("load_checkpoint", None) is not None:
        return load_model_from_config(conf, RatioClassifier)
    elif conf.model_config.get("pt_sliced_model", None) is not None:
        return load_pt_sliced_model(conf), conf.model_config.pt_sliced_model.get("model_save_path", "none")
    else:
        raise ValueError("No checkpoint provided for ratio model!")


def fakes_trainer(config: DictConfig, trial: optuna.trial.Trial | None = None) -> float | None:
    setup_analysis_dirs(config)
    log_hydra_config(config)

    if trial is not None:
        config = suggest_to_config(config, trial)

    experiment_config = config.experiment_config
    experiment_config = set_run_name(experiment_config, config.model_config)

    dataset_config, model_config = config.dataset_config, config.model_config

    experiment_name = experiment_config.experiment_name
    dataset_name = dataset_config.name
    model_name = model_config.name

    load_checkpoint = model_config.load_checkpoint
    if load_checkpoint is not None:
        model_save_file = os.path.join(model_config.training_config.model_save_path, load_checkpoint)

        if not os.path.exists(model_save_file):
            logging.warning(f"Checkpoint {load_checkpoint} does not exist! Training from scratch.")
            load_checkpoint = None
        else:
            logging.info(f"Loading from checkpoint {load_checkpoint}.")

    model: L.LightningModule
    tracker: Tracker | None

    if model_name == "numModel" or model_name == "denModel":
        if load_checkpoint:
            try:
                load_num_den_model(config)
                logging.info("Num/den model is ready to be used.")
            except Exception as e:
                logging.error(f"Error loading model: {e}")

            return None

        if config.experiment_config.plot_metrics_n_epoch is None:
            tracker = None
        else:
            tracker = NumDenTracker(
                experiment_config,
                dataset_config,
                model_config,
                config.plotting_config,
                tracker_path=os.path.join(experiment_config.tracker_path, model_name),
                num_den="num" if model_name == "numModel" else "den",
            )

        model = build_num_den_model(
            model_config,
            dataset_config,
            tracker,
            run_name=experiment_config.run_name,
        )

    elif model_name == "ratioModel":
        if load_checkpoint or model_config.get("pt_sliced_model", None) is not None:
            try:
                load_ratio_model(config)
                logging.info("Ratio model is ready to be used.")
            except Exception as e:
                logging.error(f"Error loading model: {e}")

            return None

        if config.experiment_config.plot_metrics_n_epoch is None:
            tracker = None
        else:
            tracker = RatioTracker(
                experiment_config,
                dataset_config,
                model_config,
                config.plotting_config,
                tracker_path=os.path.join(experiment_config.tracker_path, model_name),
            )

        model = build_ratio_model(config, tracker, run_name=experiment_config.run_name)

    else:
        raise NotImplementedError(f"Model {model_config.name} not implemented!")

    if trial is not None:
        logging.info("[red]Using hyperparameter tunning!")
        experiment_name = f"sweep{experiment_name[0].upper() + experiment_name[1:]}"

    logging.info(f"[bold green]Starting experiment: {experiment_name}.")

    callbacks = get_callbacks(
        model_config.training_config,
        experiment_config.run_name,
        monitor=model_config.training_config.monitor if trial is None else config.tuning_config.monitor,
        refresh_rate=experiment_config.get("refresh_rate", 100),
    )

    if trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor=config.tuning_config.monitor))

    logger = get_logger(
        experiment_name,
        experiment_config.run_name,
        experiment_config.save_dir,
        comet_api_key=experiment_config.get("comet_api_key", None),
        comet_project_name=experiment_config.get("comet_project_name", None),
    )

    T_0 = get_T_0_from_scheduler_config(model_config)
    trainer = get_trainer(experiment_config, model_config.training_config, logger, callbacks, val_check_interval=T_0)

    dm = get_fakes_data_module(dataset_config, dataset_name, model_name, config=config)

    trainer.fit(model, dm)

    logging.info("[green]Done! Returning model.")

    if trial is not None:
        for callback in trainer.callbacks:  # type: ignore
            if isinstance(callback, EarlyStopping):
                return callback.best_score.item()

    return None


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> float | None:
    setup_logger(config.min_logging_level)

    seed_everything(config.experiment_config.seed, workers=True)

    tuning_config = config.get("tuning_config", None)

    if tuning_config is not None:
        pruner_name = tuning_config.pruning

        pruner: optuna.pruners.BasePruner

        if pruner_name is None:
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = getattr(optuna.pruners, pruner_name)()

        study = optuna.create_study(
            sampler=getattr(optuna.samplers, tuning_config.sampler)(),
            study_name=tuning_config.study_name,
            storage=tuning_config.storage,
            direction=tuning_config.direction,
            pruner=pruner,
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: fakes_trainer(config, trial),  # type: ignore
            n_trials=tuning_config.n_trials,
            n_jobs=tuning_config.n_jobs,
            timeout=tuning_config.timeout,
        )

        logging.info(f"Number of finished trials: {len(study.trials)}")

        logging.info("Best trial:")
        trial = study.best_trial

        logging.info(f"Value: {trial.value}")

        logging.info("Params:")
        for key, value in trial.params.items():
            logging.info(f"{key}: {value}")

    else:
        fakes_trainer(config)

    return None


if __name__ == "__main__":
    main()
