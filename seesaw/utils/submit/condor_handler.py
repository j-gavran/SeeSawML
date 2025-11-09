import glob
import logging
import os
import subprocess
from pathlib import Path
from typing import Any


class CondorHandler:
    def __init__(self, batch_path: str, log_path: str, base_dir: str, output_destination: str | None = None) -> None:
        """A class to submit batch jobs to a HTCondor scheduler.

        Inspired by https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt.

        Attributes
        ----------
        batch_path : Path
            Path where the batch config files which are created will be stored.
        log_path : Path
            Path where the batch log files will be stored.
        base_dir : Path
            Directory in which batch job will execute its command.
        output_destination : Path | None
            Path where the output files will be stored (optional). Should be a path to an eos directory.

        Methods
        -------
        activate_testmode():
            Activate test mode: check config files in dry runs, no jobs submitted.
        deactivate_testmode():
            Deactivate test mode, enable submitting jobs.
        send_job(command: str, tag: str = "htcondor_job"):
            Submit job by creating config files (bash file and HTCondor submission file) and executing condor_submit.
        get_ml_data_files():
            Get ML data files from the environment variables.

        References
        ----------
        [1] - https://htcondor.readthedocs.io/en/latest/
        [2] - https://batchdocs.web.cern.ch/index.html

        """
        self.batch_path = Path(batch_path)
        self.log_path = Path(log_path)
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.output_destination = output_destination
        self._tag = "ml_job"

        # keywords to be used in HTCondor configuration file
        self._condor_options_dict = {
            "universe": "Universe",
            "jobflavour": "+JobFlavour",
            "project": "+MyProject",
            "runtime": "+RequestRuntime",
            "maxruntime": "+MaxRuntime",
            "memory": "Request_Memory",
            "disk": "Request_Disk",
            "cpu": "Request_CPUs",
            "gpu": "Request_GPUs",
            "requirements": "Requirements",
            "container": "+MySingularityImage",
            "containerargs": "+MySingularityArgs",
        }

        self._condor_options: dict[str, Any] = {}
        self._test_mode = False

    def activate_testmode(self) -> None:
        logging.info("Activated test mode: not submitting any jobs.")
        self._test_mode = True

    def deactivate_testmode(self) -> None:
        logging.info("Deactivated test mode: submitting jobs.")
        self._test_mode = False

    def get_ml_data_files(self) -> list[str]:
        ntuple_dir = os.environ.get("ANALYSIS_ML_NTUPLES_DIR", None)
        ml_data_dir = os.environ.get("ANALYSIS_ML_DATA_DIR", None)

        if ntuple_dir is not None and ml_data_dir is not None:
            raise ValueError("Specify either ML ntuple directory or ML data directory, not both!")

        elif ntuple_dir is not None:
            logging.info(f"Using ntuple directory: {ntuple_dir}")
            ml_ntuple_files = glob.glob(os.path.join(ntuple_dir, "**", "*.root"), recursive=True)

            if len(ml_ntuple_files) == 0:
                logging.warning("No ML ntuple files found in the specified directory!")
            else:
                logging.info(f"Found {len(ml_ntuple_files)} ntuple files in the specified directory.")

            return ml_ntuple_files

        elif ml_data_dir is not None:
            logging.info(f"Using ML data directory: {ml_data_dir}")
            ml_data_files = glob.glob(os.path.join(ml_data_dir, "*"))

            if len(ml_data_files) == 0:
                logging.warning("No ML data files found in the specified directory!")
            else:
                logging.info(f"Found {len(ml_data_files)} ML data files in the specified directory.")

            return ml_data_files

        else:
            return []

    def send_job(self, command: str, tag: str = "ml_job") -> None:
        self._tag = tag

        bash_file = self._make_bash_file(command)
        job_file = self._make_job_file(bash_file)

        if not self._test_mode:
            subprocess.call(f"condor_submit {job_file}", shell=True)

    def __setitem__(self, key: str, value: Any) -> None:
        self._condor_options[key] = value

    def _make_bash_file(self, command: str) -> Path:
        run_file = self.batch_path / f"batch_{self._tag}.sh"

        with run_file.open("w") as fr:
            fr.write(
                f"""#!/bin/sh
# {self._tag} batch run script
BASE_DIR={self.base_dir}
CURRENT_DIR=$(pwd)
pwd
ls -l
{command}
pwd
ls -l
"""
            )
        run_file.chmod(0o755)
        logging.info(f"Made run file {run_file}")

        return run_file

    def _make_job_file(self, run_file: Path) -> Path:
        batch_file = self.batch_path / f"batch_{self._tag}.job"

        if self.output_destination is not None:
            logging.info(f"Using condor output destination: {self.output_destination}")
            output_destination = f"root://eosuser.cern.ch//{self.output_destination.strip('/')}/$(ClusterId)/"
        else:
            logging.info("No condor output destination specified.")
            output_destination = "none"

        ml_data_files = self.get_ml_data_files()

        transfer_input_files = ""
        for f in ml_data_files:
            transfer_input_files += f"root://eosuser.cern.ch//{f.strip('/')}, "

        transfer_input_files += "root://eosuser.cern.ch//eos/user/j/jgavrano/batch_venv.tar.gz"

        with batch_file.open("w") as fs:
            for key, value in self._condor_options.items():
                if key in self._condor_options_dict:
                    if value is None:
                        continue

                    if key in ["jobflavour"]:
                        write_str = f'{self._condor_options_dict[key]}="{value}"\n'
                    else:
                        write_str = f"{self._condor_options_dict[key]}={value}\n"

                    fs.write(write_str)

            fs.write(
                f"""Executable           = {run_file}
Output               = {self.log_path}/stdout_{self._tag}_$(ClusterId).txt
Error                = {self.log_path}/stderr_{self._tag}_$(ClusterId).txt
Log                  = {self.log_path}/batch_{self._tag}_$(ClusterId).log
output_destination   = {output_destination}
transfer_input_files = {transfer_input_files}
MY.XRDCP_CREATE_DIR  = True

queue
"""
            )

        logging.info(f"Made job file {batch_file}")

        return batch_file
