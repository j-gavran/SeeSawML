import logging
import os
import time

from seesawml.fakes.training.run_training import run_training
from seesawml.utils.hydra_initalize import get_hydra_config
from seesawml.utils.loggers import setup_logger

TRIGGER_PT = {
    "ptBaseline": 30.0,
    "ptBaselineLow": 10.0,
    "pt10": 11.0,
    "pt12": 13.0,
    "pt14": 15.0,
    "pt15": 16.0,
    "pt17": 18.0,
    "pt18": 19.0,
    "pt20": 21.0,
    "pt22": 23.0,
    "pt24": 25.0,
    "pt26": 27.0,
    "pt28": 29.0,
    "pt60": 65.0,
    "pt70": 75.0,
    "pt80": 85.0,
    "pt100": 105.0,
    "pt120": 126.0,
    "pt140": 147.0,
    "pt160": 168.0,
    "ptUnprescaledEl": 315.0,
    "ptMax": 10000.0,
}


def run_pt_sliced_training(run_folder: str, experiment_name: str, overrides: list[str] | None = None) -> None:
    if overrides is None:
        overrides = []

    config = get_hydra_config(os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"))
    pt_slices = config.dataset_config.pt_slices

    for pt_slice in pt_slices:
        pt_split = pt_slice.split(":")
        pt_min, pt_max = TRIGGER_PT[pt_split[0]], TRIGGER_PT[pt_split[1]]
        pt_overrides = [
            f"experiment_config.run_name=+{pt_split[0]}_{pt_split[1]}",
            f"dataset_config.dataset_kwargs.pt_cut.min={pt_min}",
            f"dataset_config.dataset_kwargs.pt_cut.max={pt_max}",
            f"plotting_config.subtraction_plot.variables.el_pt.x_min={pt_min}",
            f"plotting_config.subtraction_plot.variables.el_pt.x_max={pt_max if pt_max < 1000.0 else 1000.0}",
            "plotting_config.subtraction_plot.variables.el_pt.logx=false",
        ]

        logging.info(f"[yellow]Running training for pt slice: {pt_min} - {pt_max} GeV.")
        run_training(run_folder, f"{experiment_name}_{pt_split[0]}_{pt_split[1]}", pt_overrides + overrides)


def main() -> None:
    setup_logger()

    experiment_name = time.asctime(time.localtime()).replace("  ", "_").replace(" ", "_").replace(":", "_")

    run_folder = os.path.join(os.environ["ANALYSIS_ML_RESULTS_DIR"], "pt_sliced_fakes")
    os.makedirs(run_folder, exist_ok=True)

    run_pt_sliced_training(run_folder, experiment_name)


if __name__ == "__main__":
    main()
