The installation does not require any containers or any ATLAS specific setup. It is a pure Python package that can be used in any environment with Python.

## Getting the Code

The package is aimed to be used as a submodule in other projects.

!!! note "Analysis Using SeeSawML"
    If the package has already been added as a submodule, you can skip this section and refer to the `README.md` file for usage instructions of the specific analysis. This applies for e.g. [`SeeSawAnalysis`](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawAnalysis) and [`ttHccAnalysis`](https://gitlab.cern.ch/atlas-physics/higp/bcquarks/ttHcc/histogramming/ttHccAnalysis).

To include it as a submodule, run the following command in your project directory:

```shell
git submodule add ssh://git@gitlab.cern.ch:7999/atlas-dch-seesaw-analyses/SeeSawML.git modules/SeeSawML
git submodule update --init --recursive
```

After this is done, you can start using the package in your project by writing YAML configuration files or importing the package in your Python code directly.

## Setting up the Environment

SeeSawML uses environment variables to set up paths for models and data. You can set the environment variables by sourcing
a setup script in every terminal session before using the package:

```shell
source <name_of_your_setup>.sh
```

An example is provided in the [`config.sh`](https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML/-/blob/main/config.sh?ref_type=heads) file in the root of the repository. You can modify it to fit your needs.

It should set the following environment variables:

- `ANALYSIS_ML_CODE_DIR`: Path to the root directory of the repository where the SeeSawML package is located.
- `ANALYSIS_ML_OUTPUT_DIR`: Path to the directory where output files (e.g. model weights, logs, plots) will be stored.
- `ANALYSIS_ML_CONFIG_DIR`: Path to the directory where configuration files are stored. This is usually a subdirectory in the code directory, e.g. `"${ANALYSIS_ML_CODE_DIR}/ml_config"`
- `ANALYSIS_ML_DATA_DIR`: Path to the directory where input data files (e.g. ROOT files, HDF5 files) are stored.
- `ANALYSIS_ML_RESULTS_DIR`: Path to the directory where results (e.g. plots, evaluation metrics) will be stored.
- `ANALYSIS_ML_MODELS_DIR`: Path to the directory where model weights (checkpoints) will be stored.
- `ANALYSIS_ML_LOGS_DIR`: Path to the directory where log files will be stored (`hydra`, `mlflow` and `stdout` logs).

Optionally, you can also set the following environment variables:

- `ANALYSIS_ML_VENV_PATH`: Path to the Python virtual environment to be used with the package. If not set, a new virtual environment will be created in the SeeSawML directory as `.venv` and the package will be installed there.
- `ANALYSIS_ML_NTUPLES_DIR`: Path to the directory where ntuples (starting ROOT files) will be stored. This can also be set in the YAML configuration files directly.
- `ANALYSIS_ML_PYTHON`: Python version to be used with the package, e.g. `3.11` or `3.12` (default).
- `ANALYSIS_ML_TORCH`: PyTorch installation to be used with the package, e.g. `cpu`, `cu127`, `cu128` (default) or `cu130`.
- `ANALYSIS_COLUMNAR_DEV`: Path to the [F9Columnar](https://gitlab.cern.ch/ijs-f9-ljubljana/F9Columnar) repository clone. This is optional and only needed if you want to develop the columnar analysis code. If not set, the package will use the git version specified in the `pyproject.toml` file.

!!! Note "Setting Up Virtual Environment"
    It is recommended to set up `ANALYSIS_ML_VENV_PATH`. For example: `export ANALYSIS_ML_VENV_PATH="/data0/jang/analysis/seesawml_venv"`. If not set, `uv` will create a new virtual environment in the SeeSawML directory and install the package there.

## Installing Dependencies

Once the environment variables are set, the virtual environment is created automatically with `uv` and all the required packages are installed the first time you source the setup script. The script will create a Python virtual environment at the path specified by `ANALYSIS_ML_VENV_PATH` and install all required packages listed in the `pyproject.toml` file. If the virtual environment already exists, it will simply be activated.

!!! Info "Using the Setup Script"
    The setup script will create and/or use the virtual environment specified by the `ANALYSIS_ML_VENV_PATH` environment variable.
    It will do this by running the following command:
    ```shell
    source "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML/seesawml/utils/setup/setup.sh"
    ```

!!! Warning
    Virtual environment creation and activation is done automatically and it is **not** recommended to do this step manually.

After successfully setting all the environment variables, you should get the following output every time you source the setup script:

```
Setting up ML environment with uv (Python 3.12, torch-cu128)

                             Welcome to
  _____                  _____                      __  __   _
 / ____|                / ____|                    |  \/  | | |
| (___     ___    ___  | (___     __ _  __      __ | \  / | | |
 \___ \   / _ \  / _ \  \___ \   / _` | \ \ /\ / / | |\/| | | |
 ____) | |  __/ |  __/  ____) | | (_| |  \ V  V /  | |  | | | |____
|_____/   \___|  \___| |_____/   \__,_|   \_/\_/   |_|  |_| |______|

Documentation: https://seesawml.docs.cern.ch/
Repository: https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML

 Variable                  Description               Path / Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ANALYSIS_ML_CODE_DIR      Analysis code             /data0/jang/analysis/SeeSawAnalysis
 ANALYSIS_ML_OUTPUT_DIR    Output                    /data0/jang/analysis
 ANALYSIS_ML_CONFIG_DIR    Configuration             /data0/jang/analysis/SeeSawAnalysis/ml_config
 ANALYSIS_ML_DATA_DIR      Data                      /data0/jang/columnar_analysis/data/fakes/hdf5/el
 ANALYSIS_ML_RESULTS_DIR   Results                   /data0/jang/analysis/ml_results
 ANALYSIS_ML_MODELS_DIR    Saved models              /data0/jang/analysis/ml_results/models
 ANALYSIS_ML_LOGS_DIR      Logs                      /data0/jang/analysis/ml_results/logs
 ANALYSIS_ML_NTUPLES_DIR   Ntuples                   Not set (optional)
 ANALYSIS_ML_PYTHON        Python environment        3.12
 ANALYSIS_ML_TORCH         PyTorch installation      cu128
 ANALYSIS_COLUMNAR_DEV     Columnar analysis utils   Not set (optional)

Python  3.12.11
PyTorch
  Version: 2.9.1+cu128
  CUDA:    12.8
  GPU:     NVIDIA GeForce RTX 4090

Use `track -p <PORT>` to start the MLFlow UI
```

And that is it! You are now ready to use SeeSawML in your analysis.
