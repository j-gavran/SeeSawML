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

    scaler_kwargs = dict(feature_scaling_config.scaler_params)

    numer_scaler_type = feature_scaling_config.get("numer_scaler_type", None)
    categ_scaler_type = feature_scaling_config.get("categ_scaler_type", None)

    ds_scaler = DatasetScaler(
        dataset_config.files,
        dataset_config.features,
        numer_scaler_type=numer_scaler_type,
        categ_scaler_type=categ_scaler_type,
        scaler_save_path=feature_scaling_config.save_path,
        n_max=feature_scaling_config.get("n_max", None),
        extra_hash=str(dataset_config.files),
        scaler_kwargs=scaler_kwargs,
        dataloader_kwargs=dict(dataset_config.dataloader_kwargs),
    )
    ds_scaler.feature_scale()


if __name__ == "__main__":
    main()
