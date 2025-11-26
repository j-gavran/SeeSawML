import argparse
import logging
import os
from pathlib import Path

from seesaw.utils.setup.run_setup import EXPORTED
from seesaw.utils.submit.condor_handler import CondorHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


VALID_COMMANDS = {
    "convert_signal": "seesaw.signal.dataset.hdf5_converter",
    "scale_signal": "seesaw.signal.dataset.dataset_scaling",
    "calculate_class_weights": "seesaw.signal.dataset.class_weights.py",
    "calculate_quantiles": "seesaw.signal.dataset.quantiles",
    "train_signal": "seesaw.signal.training.sig_bkg_trainer",
    "calibrate_signal": "seesaw.signal.training.calibrate",
    "onnx_signal": "seesaw.signal.models.model_to_onnx",
}


def get_job_command(job_name: str, command_args: list[str] | None = None) -> str:
    valid_command = VALID_COMMANDS[job_name]
    command_path = Path("/".join(valid_command.replace(".", "/").split("/")[1:]) + ".py")

    current_dir = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]

    job_path = current_dir / command_path
    job_str = str(job_path)

    if command_args:
        job_str += f" {' '.join(command_args)}"

    job_str = f"python {job_str}"

    return job_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit batch jobs to HTCondor. Docs: https://batchdocs.web.cern.ch/index.html."
    )
    parser.add_argument(
        "-c",
        "--command",
        required=True,
        type=str,
        choices=VALID_COMMANDS.keys(),
        help="Command to submit.",
    )
    parser.add_argument(
        "-a",
        "--command_args",
        nargs="*",
        default=None,
        type=str,
        help="Arguments for the submitted command.",
    )
    parser.add_argument(
        "-t",
        "--tag",
        default="ml_job",
        type=str,
        help="Tag for the job. Used to identify the job by name.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Flag for dry run.",
    )
    parser.add_argument(
        "--universe",
        default="vanilla",
        type=str,
        help="Universe for the job.",
    )
    parser.add_argument(
        "--cpu",
        default=1,
        type=int,
        help="Number of CPUs to use.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--memory",
        default=3000,
        type=int,
        help="Requested memory in MB.",
    )
    parser.add_argument(
        "--disk",
        default=None,
        type=int,
        help="Requested disk in KB.",
    )
    parser.add_argument(
        "-f",
        "--jobflavour",
        default=None,
        choices=["espresso", "microcentury", "longlunch", "workday", "tomorrow", "testmatch", "nextweek"],
        type=str,
        help="Job flavour for the job. Espresso = 20 minutes, microcentury = 1 hour, longlunch = 2 hours, workday = 8 hours, tomorrow = 1 day, testmatch = 3 days, nextweek = 1 week",
    )
    parser.add_argument(
        "--runtime",
        default=None,
        type=int,
        help="Runtime in seconds.",
    )
    parser.add_argument(
        "--maxruntime",
        default=None,
        type=int,
        help="Max runtime in seconds.",
    )
    parser.add_argument(
        "--requirements",
        default="AlmaLinux9",
        type=str,
        choices=["AlmaLinux9", "CentOS7"],
        help="Requirements for OS.",
    )
    parser.add_argument(
        "-o",
        "--output_destination",
        default=None,
        type=str,
        help="Condor output destination. Should be a path to an eos directory.",
    )

    args = parser.parse_args()

    if args.jobflavour is None and args.runtime is None and args.maxruntime is None:
        raise ValueError("Specify job flavour or run time!")

    # define directories
    base_dir = os.environ["ANALYSIS_ML_OUTPUT_DIR"]

    batch_dir = Path(base_dir) / "condor"
    batch_path = batch_dir / "batch"
    log_path = batch_dir / "batch_logs"

    for directory in [batch_path, log_path]:
        directory.mkdir(parents=True, exist_ok=True)

    # HTCondor options
    handler = CondorHandler(
        batch_path=str(batch_path),
        log_path=str(log_path),
        base_dir=base_dir,
        output_destination=args.output_destination,
    )

    handler["universe"] = args.universe
    handler["cpu"] = args.cpu
    handler["gpu"] = args.gpu
    handler["memory"] = args.memory
    handler["disk"] = args.disk
    handler["jobflavour"] = args.jobflavour
    handler["runtime"] = args.runtime
    handler["maxruntime"] = args.maxruntime
    handler["requirements"] = f'OpSysAndVer == "{args.requirements}"'

    # construct and submit the job command
    command = "export OMP_NUM_THREADS=1"

    # environment variables
    for export in EXPORTED.keys():
        if export == "ANALYSIS_ML_DATA_DIR":
            continue
        add_export = os.environ.get(export)
        if add_export:
            command += f"\nexport {export}={add_export}"

    command += '\nexport ANALYSIS_ML_DATA_DIR="${CURRENT_DIR}"\n'

    # virutual environment setup
    command += 'echo "Unpacking environment"\n'
    command += "mkdir venv\n"
    command += "tar -xzf batch_venv.tar.gz -C venv\n"
    command += "source venv/bin/activate\n"
    command += "conda-unpack\n"
    command += 'echo "Unpacking done"\n'

    # job path and python setup
    command += "cd ${BASE_DIR} \n"
    analysis_code_dir = os.environ["ANALYSIS_ML_CODE_DIR"]
    command += "export PYTHONPATH=${PYTHONPATH}:" + f"{analysis_code_dir}/modules/SeeSawML\n"

    # construct the job command with the specified job name and arguments
    job_command = get_job_command(args.command, command_args=args.command_args)

    # run the command as a job
    command += 'echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"\n'
    command += 'echo "Checking CUDA version..."\n'
    command += "(nvcc --version || nvidia-smi)\n"
    command += 'echo "Running job"\n'
    command += f"{job_command}\n"
    command += 'echo "Job done"\n'
    command += "cd ${CURRENT_DIR}"

    if args.dry_run:
        handler.activate_testmode()

    handler.send_job(command, args.tag)


if __name__ == "__main__":
    main()
