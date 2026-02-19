import hashlib
import logging
import os

import hydra
import lightning as L
import torch
from f9columnar.ml.dataloader_helpers import get_hdf5_columns
from f9columnar.ml.hdf5_dataloader import FullWeightedBatchType, WeightedBatchType
from f9columnar.utils.helpers import load_pickle
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from seesawml.models.calibration import Calibrator, get_calibration_wrapper
from seesawml.models.nn_modules import BaseLightningModule
from seesawml.signal.training.sig_bkg_trainer import get_signal_data_module, load_sig_bkg_model
from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.loggers import get_batch_progress, setup_logger


def _get_flat_batch(batch: WeightedBatchType, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    X, y, _, _, _ = batch

    X, y = X.to(device), y.to(device)

    return X, y


def _get_jagged_batch(
    batch: FullWeightedBatchType, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    Xs: list[torch.Tensor] = []

    for k in batch[0].keys():
        if k != "events":
            Xs.append(batch[0][k][0])

    X_events, y = batch[0]["events"][0], batch[0]["events"][1]

    if y is None:
        raise RuntimeError("Labels are None in the provided batch!")

    X_events, y = X_events.to(device), y.to(device)
    Xs = [x.to(device) for x in Xs]

    return X_events, Xs, y


def _get_class_weights(dataset_conf: DictConfig) -> dict[int, float]:
    classes = dataset_conf.get("classes", None)
    if classes is None:
        raise ValueError("No classes defined in the dataset configuration for class weights.")

    class_weights_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "class_weights")

    hash_name = "".join(str(classes)) + str(dataset_conf.files)
    class_weights_file_name = hashlib.md5(hash_name.encode()).hexdigest()

    logging.info("Loading class weights.")
    try:
        class_weights = load_pickle(os.path.join(class_weights_dir, f"{class_weights_file_name}.p"))
    except FileNotFoundError:
        logging.error(f"Class weights file not found in {class_weights_dir}. Please run calculate_class_weights first.")
        raise FileNotFoundError("Could not load class weights.")

    return class_weights["calib"]


def _fit_calibrator_iterative(
    train_dl: DataLoader,
    logits_model: torch.nn.Module,
    calibrator: Calibrator,
    events_only: bool,
    device: torch.device,
    class_weights: dict[int, float] | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[float, float]]:
    params_lst = []

    progress = get_batch_progress()
    progress.start()
    bar = progress.add_task("Processing batches", total=None)

    for dl_batch in train_dl:
        if events_only:
            flat_batch = _get_flat_batch(dl_batch, device)
            X, y = flat_batch
            with torch.no_grad():
                logits = logits_model(X)
        else:
            jagged_batch = _get_jagged_batch(dl_batch, device)
            X_events, Xs, y = jagged_batch
            with torch.no_grad():
                logits = logits_model(X_events, *Xs)

        if class_weights is not None:
            class_weights_t = torch.zeros_like(y, dtype=logits[0].dtype)

            for class_label, class_weight in class_weights.items():
                class_weights_t[y == class_label] = class_weight

        perm = torch.randperm(logits.size(0), device=device)
        logits, y = logits[perm], y[perm]

        if class_weights is not None:
            weights = class_weights_t[perm]

        calibrator.fit(logits, y, weights if class_weights is not None else None)
        params_lst.append((calibrator.params(), calibrator.ece_score()))

        progress.update(bar, advance=1)

    progress.stop()

    return _parse_params_lst(params_lst)


def _parse_params_lst(
    params_lst: list[tuple[dict[str, torch.Tensor], tuple[float, float]]],
) -> tuple[dict[str, torch.Tensor], tuple[float, float]]:
    acc_params: dict[str, list[torch.Tensor]] = {}
    acc_ece: dict[str, list[float]] = {"before": [], "after": []}

    for params, ece in params_lst:
        for k, v in params.items():
            if k not in acc_params:
                acc_params[k] = []
            acc_params[k].append(v)

        acc_ece["before"].append(ece[0])
        acc_ece["after"].append(ece[1])

    stacked_params: dict[str, torch.Tensor] = {}
    for k, v_lst in acc_params.items():
        stacked_params[k] = torch.stack(v_lst, dim=0)

    final_params: dict[str, torch.Tensor] = {}
    for k, v in stacked_params.items():
        final_params[k] = torch.mean(v, dim=0)

    ece_before_tensor = torch.tensor(acc_ece["before"])
    ece_after_tensor = torch.tensor(acc_ece["after"])

    mean_ece_before = torch.mean(ece_before_tensor).item()
    mean_ece_after = torch.mean(ece_after_tensor).item()

    return final_params, (mean_ece_before, mean_ece_after)


def _fit_calibrator_all(
    train_dl: DataLoader,
    logits_model: torch.nn.Module,
    calibrator: Calibrator,
    events_only: bool,
    device: torch.device,
    class_weights: dict[int, float] | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[float, float]]:
    logits_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []
    weights_all: list[torch.Tensor] = []

    progress = get_batch_progress()
    progress.start()
    bar = progress.add_task("Processing batches", total=None)

    for dl_batch in train_dl:
        if events_only:
            flat_batch = _get_flat_batch(dl_batch, device)
            X, y = flat_batch
            with torch.no_grad():
                logits = logits_model(X)
        else:
            jagged_batch = _get_jagged_batch(dl_batch, device)
            X, Xs, y = jagged_batch
            with torch.no_grad():
                logits = logits_model(X, *Xs)

        logits_all.append(logits)
        y_all.append(y)

        if class_weights is not None:
            class_weights_t = torch.zeros_like(y, dtype=logits_all[0].dtype)

            for class_label, class_weight in class_weights.items():
                class_weights_t[y == class_label] = class_weight

            weights_all.append(class_weights_t)

        progress.update(bar, advance=1)

    progress.stop()

    logits_all_tensor = torch.cat(logits_all, dim=0)
    y_all_tensor = torch.cat(y_all, dim=0)

    if class_weights is not None:
        weights_all_tensor = torch.cat(weights_all, dim=0)

    perm = torch.randperm(logits_all_tensor.size(0), device=device)
    logits_all_tensor, y_all_tensor = logits_all_tensor[perm], y_all_tensor[perm]

    if class_weights is not None:
        weights_all_tensor = weights_all_tensor[perm]

    calibrator.fit(logits_all_tensor, y_all_tensor, weights_all_tensor if class_weights is not None else None)

    return calibrator.params(), calibrator.ece_score()


def fit_calibrator(
    dm: L.LightningDataModule,
    module: BaseLightningModule,
    calib_config: DictConfig,
    dataset_config: DictConfig,
    events_only: bool = True,
    device: torch.device = torch.device("cpu"),
    is_multiclass: bool = True,
) -> dict[str, torch.Tensor]:
    calibrator = Calibrator(calib_config.method, is_binary=not is_multiclass, **calib_config.calibration_params)

    logits_model = module.model.eval()
    train_dl = dm.train_dataloader()

    if dataset_config.get("use_class_weights", False):
        class_weights = _get_class_weights(dataset_config)
    else:
        class_weights = None

    if calib_config.get("fit_all", False):
        params = _fit_calibrator_all(train_dl, logits_model, calibrator, events_only, device, class_weights)
    else:
        params = _fit_calibrator_iterative(train_dl, logits_model, calibrator, events_only, device, class_weights)

    calib_params, ece_score = params

    logging.info(f"Calibration ECE before: {ece_score[0]:.6f}, after: {ece_score[1]:.6f}")

    for k, v in calib_params.items():
        logging.info(f"Calibrator parameter {k}: {v}")

    return calib_params


def calibrate_model(config: DictConfig) -> None:
    setup_analysis_dirs(config)

    dataset_config, model_config = config.dataset_config, config.model_config

    if model_config.get("calibration_config", None) is None:
        raise ValueError("Provide a calibration_config in the model configuration for calibration!")

    calib_config = model_config.calibration_config

    logging.info(f"Using {calib_config.method} calibration method.")

    if dataset_config.dataloader_kwargs.batch_size is not None:
        logging.info("Setting batch size to None for calibration data loader.")
        with open_dict(dataset_config):
            dataset_config.dataloader_kwargs.batch_size = None

    model_name = model_config.name
    if model_name != "sigBkgClassifier":
        raise NotImplementedError(f"Model {model_name} not implemented!")

    columns_dct = get_hdf5_columns(dataset_config.files, resolve_path=True)
    events_only = not any(
        c in dataset_config.features and dataset_name != "events"
        for dataset_name, columns in columns_dct.items()
        for c in columns
    )

    classes = dataset_config.get("classes", None)
    if classes is not None and len(classes) > 2:
        logging.info("[yellow]Detected multiclass classification problem.")
        is_multiclass = True
    else:
        logging.info("[yellow]Detected binary classification problem.")
        is_multiclass = False

    stage_split_piles = dataset_config.stage_split_piles
    if "calib" not in stage_split_piles:
        raise ValueError("No 'calib' split found in dataset configuration for calibration!")

    dm = get_signal_data_module(
        dataset_conf=dataset_config,
        dataset_name=dataset_config.name,
        model_name=model_name,
        events_only=events_only,
        is_calibration=True,
    )

    load_checkpoint = model_config.load_checkpoint
    if load_checkpoint is None:
        raise ValueError("No checkpoint specified for loading the pre-trained model!")

    if "_calib.ckpt" in load_checkpoint:
        raise ValueError("The specified checkpoint is already a calibrated model!")

    model_save_file = os.path.join(model_config.training_config.model_save_path, load_checkpoint)

    if not model_save_file.endswith(".ckpt"):
        raise ValueError(f"Checkpoint {load_checkpoint} does not have a valid .ckpt extension!")

    if not os.path.exists(model_save_file):
        raise FileNotFoundError(f"Checkpoint file {model_save_file} not found!")

    pretrained_module, _ = load_sig_bkg_model(config, events_only=events_only, disable_compile=True)

    accelerator = config.experiment_config.accelerator

    if accelerator == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(accelerator)

    set_temperature = calib_config.get("set_temperature", None)

    if set_temperature is None:
        calib_params = fit_calibrator(
            dm,
            pretrained_module,
            calib_config,
            config.dataset_config,
            events_only=events_only,
            device=device,
            is_multiclass=is_multiclass,
        )
    else:
        if calib_config.method != "temperature":
            raise ValueError("Temperature setting can only be used with 'temperature' calibration method!")

        logging.info(f"Setting temperature to {set_temperature}.")
        calib_params = {"temperature": torch.tensor(float(set_temperature), device=device)}

    calib_wrapper = get_calibration_wrapper(calib_config.method, events_only)
    calib_model = calib_wrapper(pretrained_module.model, **calib_params)

    hyper_params = torch.load(model_save_file, map_location="cpu", weights_only=False)["hyper_parameters"]

    calib_model_saved_file = model_save_file.replace(".ckpt", "_calib.ckpt")

    logging.info(f"Saving calibrated model to {calib_model_saved_file}")
    torch.save(
        {
            "state_dict": calib_model.state_dict(),
            "hyper_parameters": hyper_params,
        },
        calib_model_saved_file,
    )


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)

    seed_everything(config.experiment_config.seed, workers=True)

    calibrate_model(config)


if __name__ == "__main__":
    main()
