import logging
import os

import hydra
from f9columnar.ml.dataloader_helpers import get_hdf5_metadata
from f9columnar.ml.dataset_scaling import DatasetScaler
from f9columnar.utils.loggers import timeit
from omegaconf import DictConfig

from seesawml.signal.utils import get_classifier_labels
from seesawml.utils.helpers import setup_analysis_dirs
from seesawml.utils.loggers import setup_logger


@timeit(unit="s")
@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    dataset_config = config.dataset_config
    feature_scaling_config = dataset_config.feature_scaling

    scaler_kwargs = dict(feature_scaling_config.scaler_params)
    dataloader_kwargs = dict(dataset_config.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null.")

    dataset_kwargs = {}

    metadata = get_hdf5_metadata(dataset_config.files, resolve_path=True)
    labels = metadata.get("labels", None)

    if labels is not None:
        classes = dataset_config.get("classes", None)
        if classes is not None:
            _, remap_labels, _ = get_classifier_labels(classes, labels)
            dataset_kwargs["remap_labels"] = remap_labels
            dataset_kwargs["max_label"] = max(labels.values())

    numer_scaler_type = feature_scaling_config.get("numer_scaler_type", None)
    categ_scaler_type = feature_scaling_config.get("categ_scaler_type", None)

    ds_scaler = DatasetScaler(
        dataset_config.files,
        dataset_config.features,
        numer_scaler_type=numer_scaler_type,
        categ_scaler_type=categ_scaler_type,
        scaler_save_path=feature_scaling_config.save_path,
        n_max=feature_scaling_config.get("n_max", None),
        extra_hash=str(config.dataset_config.files),
        scaler_kwargs=scaler_kwargs,
        dataset_kwargs=None if len(dataset_kwargs) == 0 else dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )
    ds_scaler.feature_scale()


if __name__ == "__main__":
    main()
