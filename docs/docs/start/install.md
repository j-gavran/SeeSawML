The installation does not require any containers or any ATLAS specific setup. It is a pure Python package that can be used in any environment with Python `3.10` or higher.

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
- `ANALYSIS_ML_VENV_PATH`: Path to the Python virtual environment to be used with the package. If using TNAnalysis, it will default to the TNAnalysis uv virtual environment and you do not need to set this variable.
- `ANALYSIS_ML_CONFIG_DIR`: Path to the directory where configuration files are stored. This is usually a subdirectory in the code directory, e.g. `"${ANALYSIS_ML_CODE_DIR}/ml_config"`
- `ANALYSIS_ML_DATA_DIR`: Path to the directory where input data files (e.g. ROOT files, HDF5 files) are stored.
- `ANALYSIS_ML_RESULTS_DIR`: Path to the directory where results (e.g. plots, evaluation metrics) will be stored.
- `ANALYSIS_ML_MODELS_DIR`: Path to the directory where model weights (checkpoints) will be stored.
- `ANALYSIS_ML_LOGS_DIR`: Path to the directory where log files will be stored (`hydra`, `mlflow` and `stdout` logs).
- `ANALYSIS_ML_NTUPLES_DIR`: Path to the directory where ntuples (starting ROOT files) will be stored. This can also be set in the YAML configuration files directly.

!!! Note "Setting Up Virtual Environment"
    It is recommended to set up `ANALYSIS_ML_VENV_PATH`. For example: `export ANALYSIS_ML_VENV_PATH="/data0/jang/analysis/seesawml_venv"` or
    `export ANALYSIS_ML_VENV_PATH="/afs/cern.ch/user/j/jgavrano/analysis/condor_venv"`

!!! Tip "Using TNAnalysis Virtual Environment"
    If using TNAnalysis virtual environment, you should setup the TNAnalysis environment first. Additionally you can add the following line to your setup file: `UV_CACHE_DIR`. For more information, see the [uv docs](https://docs.astral.sh/uv/concepts/cache/#cache-directory).

## Installing Dependencies

Once the environment variables are set, the virtual environment is created automatically and all the required packages are installed the first time you source the setup script. The script will create a Python virtual environment at the path specified by `ANALYSIS_ML_VENV_PATH` and install all required packages listed in the `pyproject.toml` file. If the virtual environment already exists, it will simply be activated.

!!! Info "Using the Setup Script"
    The setup script will create and/or use the virtual environment specified by the `ANALYSIS_ML_VENV_PATH` environment variable.
    It will do this by running the following command:
    ```shell
    source "${ANALYSIS_ML_CODE_DIR}/modules/SeeSawML/seesaw/utils/setup/setup.sh"
    ```
    As already mentioned, if ``ANALYSIS_ML_VENV_PATH`` is not set, and TNAnalysis is being used, the TNAnalysis uv virtual environment will be used instead.

!!! Warning
    Virtual environment creation and activation is done automatically and it is **not** recommended to do this step manually.

After successfully setting all the environment variables, you should get the following output every time you source the setup script:

```
Activating ML virtual environment at /data0/jang/analysis/seesawml_venv
Found SeeSawML installation

                             Welcome to
  _____                  _____                      __  __   _
 / ____|                / ____|                    |  \/  | | |
| (___     ___    ___  | (___     __ _  __      __ | \  / | | |
 \___ \   / _ \  / _ \  \___ \   / _` | \ \ /\ / / | |\/| | | |
 ____) | |  __/ |  __/  ____) | | (_| |  \ V  V /  | |  | | | |____
|_____/   \___|  \___| |_____/   \__,_|   \_/\_/   |_|  |_| |______|

     https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML

Using analysis code directory: /data0/jang/analysis/SeeSawAnalysis
Using output directory: /data0/jang/analysis
ANALYSIS_ML_NTUPLES_DIR is not set!
Using configuration from: /data0/jang/analysis/SeeSawAnalysis/ml_config
Using data from: /data0/jang/analysis/ml_data/seesaw_2L
Using results directory: /data0/jang/analysis/ml_results
Using saved models directory: /data0/jang/analysis/ml_results/models
Using logs directory: /data0/jang/analysis/ml_results/logs
Using Python: 3.12.11
Checking PyTorch installation...
Using PyTorch: 2.8.0+cu128
Found GPU: NVIDIA GeForce RTX 4090
Use `track -p <PORT>` to start the MLFlow UI
```

Or if running on `lxplus`:

```
Activating ML virtual environment at /afs/cern.ch/user/j/jgavrano/analysis/condor_venv
Found SeeSawML installation

                             Welcome to
  _____                  _____                      __  __   _
 / ____|                / ____|                    |  \/  | | |
| (___     ___    ___  | (___     __ _  __      __ | \  / | | |
 \___ \   / _ \  / _ \  \___ \   / _` | \ \ /\ / / | |\/| | | |
 ____) | |  __/ |  __/  ____) | | (_| |  \ V  V /  | |  | | | |____
|_____/   \___|  \___| |_____/   \__,_|   \_/\_/   |_|  |_| |______|

     https://gitlab.cern.ch/atlas-dch-seesaw-analyses/SeeSawML

Using analysis code directory: /afs/cern.ch/user/j/jgavrano/analysis/SeeSawAnalysis
Using output directory: /afs/cern.ch/user/j/jgavrano/analysis
ANALYSIS_ML_NTUPLES_DIR is not set!
Using configuration from: /afs/cern.ch/user/j/jgavrano/analysis/SeeSawAnalysis/ml_config
Using data from: /eos/user/j/jgavrano/ml_ntuples
Using results directory: /afs/cern.ch/user/j/jgavrano/analysis/ml_results
Using saved models directory: /afs/cern.ch/user/j/jgavrano/analysis/ml_results/models
Using logs directory: /afs/cern.ch/user/j/jgavrano/analysis/ml_results/logs
Using Python: 3.12.9
Running on lxplus...
For HTCondor submission check `condor_ml_sub --help`
```

!!! Note "Using on lxplus"
    If running on `lxplus` only the SeeSawML will be installed in the virtual environment located on `afs`. The actual
    virtual environment is already created and zipped on `eos` and it will be used as is when running condor jobs. This is to avoid long installation times on `eos` and low `afs` quotas. For more details, see the [lxplus instructions](../start/lxplus.md).

And that is it! You are now ready to use SeeSawML in your analysis.
