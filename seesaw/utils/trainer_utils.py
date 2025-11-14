import logging
import os
import time

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.loggers import CometLogger, Logger, MLFlowLogger
from omegaconf import DictConfig, open_dict

from seesaw.utils.loggers import CustomRichProgressBar


def set_run_name(experiment_conf: DictConfig, model_config: DictConfig) -> DictConfig:
    if experiment_conf.get("run_name", None) is not None and not experiment_conf.run_name.startswith("+"):
        logging.info(f"[bold green]Run name: {experiment_conf.run_name}.")
        return experiment_conf

    model_type = model_config.architecture_config.model

    run_name = f"{model_config.name}_{model_type}_{time.asctime(time.localtime())}"
    run_name = run_name.replace("  ", "_").replace(" ", "_").replace(":", "_")

    with open_dict(experiment_conf):
        if experiment_conf.get("run_name", None) is not None and experiment_conf.run_name.startswith("+"):
            experiment_conf.run_name = f"{run_name}_{experiment_conf.run_name[1:]}"
        else:
            experiment_conf.run_name = run_name

    logging.info(f"[bold green]Run name: {run_name}.")

    return experiment_conf


def get_callbacks(
    training_conf: DictConfig,
    run_name: str,
    monitor: str = "val_loss",
    tqdm_refresh_rate: int = 100,
) -> list[L.Callback]:
    logging.info(f"Monitoring {monitor} for early stopping and model checkpointing.")

    file_name = run_name + "_{epoch}_{val_loss:.3f}"

    model_save_path = training_conf.model_save_path

    os.makedirs(model_save_path, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=(
                training_conf.max_epochs
                if training_conf.early_stop_patience is None
                else training_conf.early_stop_patience
            ),
        ),
        ModelCheckpoint(
            dirpath=model_save_path,
            filename=file_name,
            save_weights_only=True,
            mode="min",
            monitor=monitor,
            save_top_k=training_conf.get("save_top_k", 3),
            save_last=True,
        ),
        CustomRichProgressBar(refresh_rate=tqdm_refresh_rate),
    ]
    return callbacks


def disable_mlflow_tracking() -> None:
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["DO_NOT_TRACK"] = "true"


def get_logger(
    experiment_name: str,
    run_name: str,
    save_dir: str,
    comet_api_key: str | None = None,
    comet_project_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> Logger:
    if comet_api_key is not None:
        logging.info("[green]Using Comet logger!")
        comet_logger = CometLogger(
            api_key=comet_api_key,
            project_name=comet_project_name,
            experiment_name=run_name,
        )
        setattr(comet_logger, "_run_name", run_name)

        return comet_logger
    else:
        logging.info("[green]Using MLflow logger!")
        disable_mlflow_tracking()

        os.makedirs(save_dir, exist_ok=True)

        if mlflow_tracking_uri is None:
            mlflow_tracking_uri = f"sqlite:///{save_dir}/mlflow.sqlite"
            logging.info(f"Setting MLflow tracking URI to {mlflow_tracking_uri}.")

        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=mlflow_tracking_uri,
            artifact_location=save_dir,
            log_model=False,
        )

        return mlf_logger


def get_trainer(
    experiment_conf: DictConfig,
    training_conf: DictConfig,
    logger: Logger,
    callbacks: list[L.Callback],
    val_check_interval: int | float | None = None,
) -> L.Trainer:
    """Set up the PyTorch Lightning trainer.

    See: https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api

    """
    precision = experiment_conf.get("precision", "32-true")

    if "32" in str(precision):
        float32_matmul_precision = experiment_conf.get("float32_matmul_precision", "high")
        torch.set_float32_matmul_precision(float32_matmul_precision)

    if val_check_interval is None:
        check_val_every_n_epoch = experiment_conf.check_eval_n_epoch
    else:
        logging.info(f"Setting val_check_interval to {val_check_interval}.")
        check_val_every_n_epoch = None

    trainer = L.Trainer(
        max_epochs=training_conf.max_epochs,
        accelerator=experiment_conf.accelerator,
        devices=experiment_conf.devices,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=experiment_conf.log_every_n_steps,
        num_sanity_val_steps=experiment_conf.num_sanity_val_steps,
        val_check_interval=val_check_interval,
        logger=logger,
        callbacks=callbacks,
        precision=precision,
        gradient_clip_val=training_conf.get("gradient_clip_val", None),
    )

    return trainer


def get_T_0_from_scheduler_config(model_conf: DictConfig) -> int | None:
    scheduler_config = model_conf.training_config.get("scheduler", None)

    if scheduler_config is None:
        return None

    if scheduler_config.get("interval", "epoch") != "step":
        return None

    max_epochs = model_conf.training_config.max_epochs
    scheduler_params = scheduler_config.get("scheduler_params", {})

    if scheduler_config.scheduler_name == "CosineAnnealingLR" or scheduler_config.scheduler_name == "CosineWarmup":
        T_max = scheduler_params.get("T_max", None)
        if T_max is None:
            T_max = max_epochs
            logging.info(f"Setting T_max to {T_max} based on max_epochs.")
            with open_dict(model_conf):
                model_conf.training_config.scheduler.scheduler_params.T_max = T_max

        T_0 = T_max
    elif scheduler_config.scheduler_name == "CosineAnnealingWarmRestarts":
        T_0 = scheduler_params.get("T_0", None)
        if T_0 is None:
            T_0 = max_epochs // 2
            logging.info(f"Setting T_0 to {T_0} based on max_epochs.")
            with open_dict(model_conf):
                model_conf.training_config.scheduler.scheduler_params.T_0 = T_0
    else:
        return None

    return T_0
