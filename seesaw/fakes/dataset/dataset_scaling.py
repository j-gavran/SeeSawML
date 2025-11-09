import os

import hydra
from f9columnar.ml.dataset_scaling import DatasetScaler
from omegaconf import DictConfig

from seesaw.utils.helpers import setup_analysis_dirs
from seesaw.utils.loggers import setup_logger


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig) -> None:
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    dataset_config = config.dataset_config
    feature_scaling_config = dataset_config.feature_scaling

    scaler_kwargs = dict(feature_scaling_config)

    scaler_kwargs.pop("scaler_type", None)
    scaler_kwargs.pop("save_path", None)
    scaler_kwargs.pop("n_max", None)
    extra_hash = str(dataset_config.files)

    ds_scaler = DatasetScaler(
        dataset_config.files,
        feature_scaling_config.scaler_type,
        dataset_config.features,
        scaler_save_path=feature_scaling_config.save_path,
        n_max=feature_scaling_config.get("n_max", None),
        extra_hash=extra_hash,
        scaler_kwargs=scaler_kwargs,
        dataloader_kwargs=dict(dataset_config.dataloader_kwargs),
    )
    ds_scaler.feature_scale()


if __name__ == "__main__":
    main()
