import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch

from seesawml.models.activations import softabs


def get_ratio(f: np.ndarray) -> np.ndarray:
    return np.exp(f)


def get_weights(r: np.ndarray, mc_label: int, data_label: int, is_data: bool = True) -> np.ndarray:
    if mc_label == 1 and data_label == 0:
        if is_data:
            w = 1.0 - r
        else:
            w = 1.0 / r - 1.0

    elif mc_label == 0 and data_label == 1:
        if is_data:
            w = 1.0 - 1.0 / r
        else:
            w = r - 1.0

    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot logit regions with Softabs.")
    parser.add_argument(
        "--is_data",
        action="store_true",
        help="If true, plot data prescale weights. Otherwise, plot MC weights.",
    )
    args = parser.parse_args()

    is_data = args.is_data

    f = np.linspace(-2, 2, 1000)
    r = get_ratio(f)
    w01 = get_weights(r, mc_label=0, data_label=1, is_data=is_data)
    w10 = get_weights(r, mc_label=1, data_label=0, is_data=is_data)

    sa = softabs(torch.Tensor(f)).numpy()

    hep.style.use("ATLAS")

    plt.figure(figsize=(7, 5.8))

    if is_data:
        plt.plot(f, r, label="r")
        plt.plot(f, w01, label=r"$1 - 1/r$ for MC=0, data=1")
        # plt.plot(f, w10, label=r"$1 - r$ for MC=1, data=0")
    else:
        plt.plot(f, r, label="r")
        plt.plot(f, w01, label=r"$1/r - 1$ for MC=0, data=1")
        plt.plot(f, w10, label=r"$r - 1$ for MC=1, data=0")

    plt.axvline(0, c="k", alpha=0.2)
    plt.axhline(0, c="k", alpha=0.2)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.yticks([-2, -1, 0, 1, 2])
    plt.xticks([-2, -1, 0, 1, 2])

    plt.fill_between([0, 2], y1=0, y2=2, alpha=0.05, color="g")
    plt.fill_between([-2, 0], y1=0, y2=2, alpha=0.02, color="r")
    plt.fill_between([-2, 0], y1=-2, y2=0, alpha=0.02, color="r")
    plt.fill_between([0, 2], y1=-2, y2=0, alpha=0.02, color="r")

    plt.plot([0, 2], [0, 0], c="k", lw=2, ls="--", alpha=0.3, zorder=10)
    plt.plot([0, 0], [0, 2], c="k", lw=2, ls="--", alpha=0.3, zorder=10)

    if is_data:
        plt.axhline(1, c="k", alpha=0.2)
    else:
        plt.axhline(-1, c="k", alpha=0.2)

    plt.plot(f, sa, c="C2", label="Softabs")

    plt.xlabel("$q$ (logit output)")

    if is_data:
        plt.ylabel("Data weights")
    else:
        plt.ylabel("MC weights")

    plt.legend(loc="lower right")
    plt.tight_layout()

    save_dir = f"{os.environ['ANALYSIS_ML_RESULTS_DIR']}/logit_regions"
    os.makedirs(save_dir, exist_ok=True)

    if is_data:
        plt.savefig(f"{save_dir}/logit_regions_data.pdf")
    else:
        plt.savefig(f"{save_dir}/logit_regions_mc.pdf")
