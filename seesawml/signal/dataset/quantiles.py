import hashlib
import logging
import os
from typing import Any

import hydra
import numpy as np
from f9columnar.ml.dataloader_helpers import get_hdf5_metadata
from f9columnar.ml.hdf5_dataloader import get_ml_hdf5_dataloader
from f9columnar.utils.helpers import dump_pickle
from f9columnar.utils.loggers import timeit
from omegaconf import DictConfig
from pytdigest import TDigest

from seesawml.signal.utils import get_classifier_labels
from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.loggers import get_batch_progress, setup_logger


@timeit(unit="s")
@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    dataset_conf = config.dataset_config

    feature_scaling_config = dataset_conf.get("feature_scaling", None)

    if feature_scaling_config is not None:
        feature_scaling_kwargs = {
            "numer_scaler_type": feature_scaling_config.get("numer_scaler_type", None),
            "categ_scaler_type": feature_scaling_config.get("categ_scaler_type", None),
            "scaler_path": feature_scaling_config.get("save_path", None),
            "scalers_extra_hash": str(dataset_conf.files),
        }
    else:
        logging.info("Feature scaling is disabled.")

    ple_bins = dataset_conf.get("ple_bins", None)
    if ple_bins is None:
        raise ValueError("ple_bins must be specified in feature_scaling config.")
    else:
        ple_bins = int(ple_bins)

    dataloader_kwargs = dict(config.dataset_config.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null.")

    dataset_kwargs: dict[str, Any] = {}
    dataset_kwargs = dataset_kwargs | feature_scaling_kwargs
    dataset_kwargs["drop_last"] = False

    metadata = get_hdf5_metadata(dataset_conf.files, resolve_path=True)
    labels = metadata.get("labels", None)

    if labels is not None:
        classes = dataset_conf.get("classes", None)
        if classes is not None:
            _, remap_labels, _ = get_classifier_labels(classes, labels)
            dataset_kwargs["remap_labels"] = remap_labels
            dataset_kwargs["max_label"] = max(labels.values())

    logging.info("[green]Initializing dataloaders...[/green]")
    hdf5_dataloader, selection, _ = get_ml_hdf5_dataloader(
        "quantiles",
        config.dataset_config.files,
        config.dataset_config.features,
        stage_split_piles=None,
        stage=None,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    dataset_keys = list(selection.keys())

    quantile_digests: dict[str, list[TDigest]] = {}
    for ds_key in dataset_keys:
        quantile_digests[ds_key] = []
        for _ in selection[ds_key].offset_numer_columns_idx:
            quantile_digests[ds_key].append(TDigest())

    progress = get_batch_progress()
    progress.start()
    bar = progress.add_task("Processing batches", total=None)

    total_events = 0

    for batch in hdf5_dataloader:
        ds_batch, _ = batch
        total_events += ds_batch[dataset_keys[0]].X.shape[0]

        for ds_key in dataset_keys:
            numer_idx = selection[ds_key].offset_numer_columns_idx
            X = ds_batch[ds_key].X[..., numer_idx].numpy()

            for f_i in range(X.shape[-1]):
                x_i = X[..., f_i]

                if x_i.ndim != 1:
                    x_i = x_i.flatten()

                pad_value = selection[ds_key].pad_value
                if pad_value is not None:
                    x_i = x_i[x_i != pad_value]

                quantile_digests[ds_key][f_i].update(x_i)

        progress.update(bar, advance=1)

    progress.stop()
    logging.info(f"Processed a total of {total_events} events.")

    interval = 1 / ple_bins
    bins_range = np.arange(0.0, 1 + interval, interval)

    quantile_bins_dct: dict[str, list[np.ndarray]] = {}
    for ds_key in dataset_keys:
        quantile_bins_dct[ds_key] = []
        for digest in quantile_digests[ds_key]:
            bins = digest.inverse_cdf(bins_range).astype(np.float32)

            bins = np.concatenate(([-np.inf], bins, [np.inf]))

            bins = np.unique(bins)
            quantile_bins_dct[ds_key].append(bins)

    for ds_key in dataset_keys:
        feature_names = selection[ds_key].numer_columns
        features_names = [feature_names[i] for i in selection[ds_key].offset_numer_columns_idx]

        log_str = f"Quantile bins for {ds_key}:\n"

        for f_i, bins in enumerate(quantile_bins_dct[ds_key]):
            f_i_name = features_names[f_i]
            round_bins = [float(f"{b:.2e}") for b in bins]
            log_str += f"{f_i_name}: {round_bins[1:-1]} (n_edges={len(bins)})\n"

        logging.info(log_str[:-1])

    quantile_bins_dir = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "quantile_bins")
    os.makedirs(quantile_bins_dir, exist_ok=True)

    hash_name = str(config.dataset_config.files) + str(sorted(config.dataset_config.features)) + str(ple_bins)
    quantile_bins_file_name = hashlib.md5(hash_name.encode()).hexdigest()

    quantile_bins_file_path = os.path.join(quantile_bins_dir, f"{quantile_bins_file_name}.p")
    logging.info(f"Saving quantile bins to {quantile_bins_file_path}.")
    dump_pickle(quantile_bins_file_path, quantile_bins_dct)


if __name__ == "__main__":
    main()
