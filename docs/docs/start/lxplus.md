!!! Warning "lxplus Support"
    Not yet available! This section will be updated once support for `lxplus` is implemented.

## GPU support

GPU support on `lxplus` is available through `ssh <user>@lxplus-gpu.cern.ch`.

## Condor support

Alternatively to running ML locally, `lxplus` can also be used on `ssh <user>@lxplus.cern.ch`.

Jobs can be submitted with `condor_ml_sub <condor_args> -c <command> -a <command_args>` command. To get help for all the available arguments, use `condor_ml_sub --help`.

To monitor a running job, you can use `condor_q` to check its status. Additionally, if you want to access the terminal of the machine where the job is currently executing, use the following command:

```shell
condor_ssh_to_job -auto-retry <job_id>
```

This will open a terminal session directly into the execution environment of the job, allowing you to interact with it in real time.

!!! Example "A Typical Workflow on lxplus"
    An example of a typical workflow on `lxplus` is described below.

    Start by setting (replace with your own paths):

    ```shell
    export ANALYSIS_ML_CODE_DIR="/afs/cern.ch/user/j/jgavrano/analysis/SeeSawAnalysis"
    export ANALYSIS_ML_OUTPUT_DIR="/afs/cern.ch/user/j/jgavrano/analysis"
    export ANALYSIS_ML_VENV_PATH="/afs/cern.ch/user/j/jgavrano/analysis/condor_venv"
    ```

    1. Convert ROOT files to HDF5 files:

        Set `ANALYSIS_ML_NTUPLES_DIR` to an `eos` path where ROOT files can be found and run:
        ```shell
        condor_ml_sub --command convert_signal --maxruntime 10800 --output_destination /eos/user/j/jgavrano/ --cpu 8
        ```
        Converted HDF5 files will get saved to `--output_destination` inside a directory with job id (you can rename this later).

    2. Perform feature normalization to get the scalers:

        Set `ANALYSIS_ML_DATA_DIR` to where HDF5 files from `convert_signal` can be found and run:
        ```shell
        condor_ml_sub --command scale_signal --maxruntime 3600 --cpu 8
        ```

    3. Train a model:

        ```shell
        condor_ml_sub --command train_signal --maxruntime 28800 --cpu 10 --gpu 1
        ```

    4. Convert to onnx:

        Set `ANALYSIS_ML_DATA_DIR` and then run:
        ```shell
        condor_ml_sub --command onnx_signal --maxruntime 1800 --cpu 8
        ```

    The arguments passed to the above commands are not optimal and should be adjusted based on the dataset size, model complexity, and available resources.

!!! Note "Requesting Multiple GPUs"
    You can request multiple GPUs per job by setting `--gpu <num_gpus>` (up to 5). However, make sure that your job actually needs that many GPUs, otherwise the job might get stuck in the queue for a long time waiting for resources to become available.

### Setting up Comet logger

MLFlow is not supported when running on lxplus. An alternative is to use Comet ML for experiment tracking and logging.
To make an account go to [https://www.comet.com/site/](https://www.comet.com/site/). After registering you will get an API key that you can add to `training_config.yaml`:

```yaml
experiment_config:
  comet_api_key: <key>
  comet_project_name: null
```

After this is set, you can monitor training from the web. If you want a custom project name, you can make it from the comet website by clicking on `+ New project` and specifying it in the `comet_project_name` field.
