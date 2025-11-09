The package is configured using multiple YAML files that define various settings and options. The final configuration is
a combination of these files handled by [Hydra](https://hydra.cc/docs/intro/), allowing for flexible and composable configurations.

## Configuration File Structure

The configuration files are located in the `ANALYSIS_ML_CONFIG_DIR` directory. There are two top level folders available
in this directory: `signal/` and `fakes/`. Each folder contains YAML files that define specific configurations for different aspects of the package.
`signal/` contains configurations related to signal vs background classification, while `fakes/` contains configurations for fake lepton estimation.

Main entry point configuration files for both `signal/` and `fakes/` are `convert_config.yaml` and `training_config.yaml`.
First one is used for data conversion and preprocessing, while the second one is used for model training and evaluation.

HDF5 dataset configuration is imported from `hdf5_config/` folder into the `convert_config.yaml` files. This folder contains YAML files that define the structure and features for HDF5 datasets used in the analysis.

Model, training dataset, plotting and tuning configuration are imported from `model_config/`, `dataset_config/`, `plotting_config/` and `tuning_config/` folders into the `training_config.yaml` files.

!!! Info
    In the `convert_config.yaml` file located in `signal/` or `fakes/`, the HDF5 dataset configuration is imported as follows:

    ```yaml
    defaults:
      - _self_
      - hdf5_config: ... # HDF5 config to use, set to one from hdf5_config directory

      - override hydra/hydra_logging: disabled # disabled by default
      - override hydra/job_logging: disabled # disabled by default
    ```

    Similarly, in the `training_config.yaml` file, model, dataset, and plotting configurations are imported:

    ```yaml
    defaults:
      - _self_
      - dataset_config: ... # change to training dataset from dataset_config directory
      - model_config: ... # change to model config from model_config directory
      - plotting_config: plotting_config # there is only one plotting config for now

      - override hydra/hydra_logging: disabled
      - override hydra/job_logging: disabled
    ```

!!! Note
    Each main configuration file (`convert_config.yaml` and `training_config.yaml`) should start with:
    ```yaml
    min_logging_level: info

    # this should not be changed
    hydra:
      output_subdir: null
      run:
        dir: .
    ```
    The `min_logging_level` sets the minimum logging level for the package, while the `hydra` section configures Hydra to output logs and results in the current working directory.

!!! Example
    Example directory structure from the [SeeSawAnalysis](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawAnalysis/-/tree/main/ml_config?ref_type=heads) of the configuration files:
    ```
    |   ├── fakes
    │   ├── convert_config.yaml
    │   ├── dataset_config
    │   │   ├── opendata_electrons.yaml
    │   │   ├── seesaw_electrons.yaml
    │   │   ├── seesaw_muons.yaml
    │   │   └── toy_dataset.yaml
    │   ├── hdf5_config
    │   │   ├── opendata_electrons.yaml
    │   │   ├── seesaw_electrons.yaml
    │   │   ├── seesaw_muons.yaml
    │   │   └── toy_dataset.yaml
    │   ├── model_config
    │   │   ├── den.yaml
    │   │   ├── num.yaml
    │   │   └── ratio.yaml
    │   ├── opendata_SR.yaml
    │   ├── plotting_config
    │   │   ├── closure_plot.yaml
    │   │   ├── density_plot.yaml
    │   │   ├── model_ff_plot.yaml
    │   │   ├── plotting_config.yaml
    │   │   ├── subtraction_plot.yaml
    │   │   └── toy_plot.yaml
    │   ├── training_config.yaml
    │   └── tuning_config
    │       └── optuna.yaml
    └── signal
        ├── convert_config.yaml
        ├── dataset_config
        │   ├── lrsm.yaml
        │   ├── ttHcc_2L.yaml
        │   ├── typeIIIseesaw.yaml
        │   ├── typeIIseesaw_2L.yaml
        │   ├── typeIIseesaw_3L.yaml
        │   └── typeIIseesaw_4L.yaml
        ├── hdf5_config
        │   ├── lrsm.yaml
        │   ├── ttHcc_2L.yaml
        │   ├── typeIIIseesaw.yaml
        │   ├── typeIIseesaw_2L.yaml
        │   ├── typeIIseesaw_3L.yaml
        │   └── typeIIseesaw_4L.yaml
        ├── model_config
        │   ├── event_transformer.yaml
        │   ├── jagged_transformer.yaml
        │   ├── mlp.yaml
        │   └── resnet.yaml
        ├── plotting_config
        │   ├── plotting_config.yaml
        │   ├── score_plot.yaml
        │   └── signal_plot.yaml
        └── training_config.yaml
    ```
    As depicted above, each main folder contains its respective configuration files and subdirectories for different configuration aspects.

!!! Warning
    Hyperparameter tuning with [Optuna](https://optuna.org/) is currently only supported for fakes analyses and not for signal vs background classification. See issue [here](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML/-/work_items/1).
