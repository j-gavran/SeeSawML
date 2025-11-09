After sourcing the setup script and setting all the environment variables correctly, you can start using SeeSawML in your analysis.
The workflow typically involves creating (or changing) configuration files in YAML format to define your machine learning tasks, datasets, and model parameters. You can then run analysis scripts that utilize SeeSawML functionalities by typing commands in your terminal.

## List of Available Commands

SeeSawML provides several command-line tools to facilitate different tasks. They are defined in the [`pyproject.toml`](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML/-/blob/main/pyproject.toml?ref_type=heads) file under the `[project.scripts]` section. A comprehensive list of available commands is listed below split into different categories.

### Setup and Submission

- `seesaw_ml_setup`: Sets up the environment for SeeSawML package.
- `condor_ml_sub`: Submits ML training jobs to the HTCondor batch system from `lxplus`.
- `download_open_data`: Downloads public ATLAS Open Data for testing and development.

### General Utilities

- `pile_stats`: Computes and saves statistics about the HDF5 dataset.
- `plot_piles`: Plot all distributions of features in the HDF5 dataset.

!!! Tip
    It is good practice to run `pile_stats` and `plot_piles` after data conversion to verify the dataset quality.

### Signal vs Background Classification

- `plot_signal`: Plot signal vs background distributions. Can also plot model output distributions.
- `convert_signal`: Convert ROOT files to HDF5 format for signal vs background dataset.
- `scale_signal`: Feature scaling.
- `calculate_class_weights`: Calculate class weights for imbalanced datasets.
- `calculate_quantiles`: Calculate feature quantiles for feature transformation.
- `train_signal`: Train signal vs background classification model.
- `onnx_signal`: Export trained model and metadata to ONNX format.

!!! Example
    A common workflow for training a signal vs background classification model would involve running the following commands in sequence:
    ```shell
    convert_signal
    scale_signal
    calculate_class_weights
    train_signal
    plot_signal
    onnx_signal
    ```

!!! Note
    Each command reads its configuration from a YAML file using [Hydra](https://hydra.cc/docs/intro/) so you can customize the behavior of each step by modifying the corresponding configuration file, usually located in the directory specified by the `ANALYSIS_ML_CONFIG_DIR` environment variable.

!!! Tip
    You can also change the configuration directly from the command line without modifying the YAML files. For example, to change the number of training epochs you can run: `train_signal model_config.training_config.max_epochs=50`.

### Fakes Estimation

- `plot_fakes`: Plot fake lepton estimation distributions.
- `convert_fakes`: Convert ROOT files to HDF5 format for fake lepton estimation dataset.
- `convert_toy_fakes`: Make toy fake lepton dataset.
- `plot_toy_fakes`: Plot toy fake lepton dataset distributions.
- `scale_fakes`: Feature scaling.
- `train_fakes_model`: Train a single model.
- `train_fakes`: Train numerator, denominator and ratio models in sequence.
- `train_pt_sliced_fakes`: Same as `train_fakes` but trains models in pT slices.
- `fakes_closure`: Perform closure test for fake lepton estimation.
- `plot_fakes_model_ff`: Plot fake factors.
- `plot_fakes_density_ratio`: Plot density ratios.
- `fakes_opendata_SR`: Perform fake lepton estimation on ATLAS Open Data.
- `onnx_fakes`: Export trained model and metadata to ONNX format.
