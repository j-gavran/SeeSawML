import hist
import matplotlib.pyplot as plt
import numpy as np
from f9columnar.utils.helpers import handle_plot_exception
from matplotlib.backends.backend_pdf import PdfPages
from plothist import plot_comparison, plot_error_hist, plot_hist
from uncertainties import unumpy as unp

from seesaw.utils.features import get_feature
from seesaw.utils.plots_utils import atlas_label


@handle_plot_exception
def plot_num_den_weights(
    weights_hists: dict[str, hist.Hist], save_dir: str, save_prefix: str, atlas_marker: str = "Internal"
) -> None:
    labels = {
        "data_prescales": "Data",
        "data_out": "NN data output",
        "data_density": "NN data density",
        "data_sub": "Data - MC for data",
        "data_reweighted": "Data reweighted",
        "mc_weights": "MC",
        "mc_out": "NN MC output",
        "mc_density": "NN MC density",
        "mc_sub": "Data - MC for MC",
        "mc_reweighted": "MC reweighted",
    }

    figs = []
    for k, label in labels.items():
        fig, ax = plt.subplots(figsize=(7, 5.8))

        h = weights_hists[k]
        plot_hist(h, ax=ax, histtype="step", lw=2, color="C0")

        label = labels[k]

        ax.set_xlabel(f"{label} weights")
        ax.set_ylabel("Events")
        ax.set_yscale("log")

        ax.set_ylim(1, None)

        if atlas_marker is not None:
            atlas_label(ax, loc=0, llabel=atlas_marker)

        figs.append(fig)
        plt.close(fig)

    with PdfPages(f"{save_dir}/{save_prefix}weights.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


@handle_plot_exception
def plot_num_den_hists(
    hists: dict[str, dict[str, hist.Hist]],
    num_den: str,
    save_dir: str,
    save_prefix: str,
    atlas_marker: str = "Internal",
) -> None:
    figs = []

    for k, h_dct in hists.items():
        data_h, mc_h = h_dct["data"], h_dct["mc"]
        data_reweighted_h, mc_reweighted_h = h_dct["data_reweighted"], h_dct["mc_reweighted"]

        mc_h_unp = unp.uarray(mc_h.values(), np.sqrt(mc_h.variances()))  # type: ignore
        data_h_unp = unp.uarray(data_h.values(), np.sqrt(data_h.variances()))  # type: ignore
        sub_h_unp = data_h_unp - mc_h_unp

        sub_h = hist.Hist.new.Variable(mc_h.axes[0].edges).Weight()
        sub_h.view().value = unp.nominal_values(sub_h_unp)  # type: ignore
        sub_h.view().variance = unp.std_devs(sub_h_unp) ** 2  # type: ignore

        for log_y in [True, False]:
            for p in range(2):
                fig, (ax_main, ax_comparison) = plt.subplots(
                    2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
                )

                if num_den == "num":
                    fig.suptitle("Numerator (tight)")
                else:
                    fig.suptitle("Denominator (loose)")

                # data
                plot_error_hist(
                    data_h,
                    label="Data",
                    ax=ax_main,
                    color="k",
                )
                # mc
                plot_error_hist(
                    mc_h,
                    label="MC",
                    ax=ax_main,
                    color="C2",
                )
                # binned data - mc
                plot_hist(
                    sub_h,
                    label="Binned Data - MC",
                    ax=ax_main,
                    histtype="step",
                    linewidth=2,
                    color="C0",
                )
                # nn reweighted
                plot_hist(
                    data_reweighted_h if p == 0 else mc_reweighted_h,
                    label="NN-reweighted data" if p == 0 else "NN-reweighted MC",
                    ax=ax_main,
                    histtype="step",
                    linewidth=2,
                    color="C1",
                )
                # comparison
                plot_comparison(
                    sub_h,
                    data_reweighted_h if p == 0 else mc_reweighted_h,
                    ax=ax_comparison,
                    comparison_ylim=[0.5, 1.5],
                    color="k",
                )

                handles, labels = ax_main.get_legend_handles_labels()

                order = ["Data", "MC", "Binned Data - MC", "NN-reweighted data" if p == 0 else "NN-reweighted MC"]

                ordered_handles = [handles[labels.index(l)] for l in order if l in labels]
                ordered_labels = [l for l in order if l in labels]

                ax_main.legend(ordered_handles, ordered_labels)
                ax_main.tick_params(labelbottom=False)
                ax_main.set_ylabel("Events")

                ax_comparison.set_xlabel(str(get_feature(k)))

                if getattr(data_h, "logx", False):
                    ax_main.set_xscale("log")
                    ax_comparison.set_xscale("log")

                ax_main.set_xlim(mc_h.axes[0].edges[0], mc_h.axes[0].edges[-1])
                ax_main.set_ylim(0, None)

                ax_comparison.set_ylabel("Binned / NN")

                if atlas_marker is not None:
                    atlas_label(ax_main, loc=0, llabel=atlas_marker)

                if log_y:
                    ax_main.set_yscale("log")
                    ax_main.set_ylim(0.01, data_h.values().max() * 10)

                if "eta" in k:
                    ax_main.set_ylim(None, data_h.values().max() * 1.5)

                figs.append(fig)
                plt.close(fig)

    with PdfPages(f"{save_dir}/{save_prefix}nn_reweighted.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


def plot_ratio_distributions(
    hists: dict[str, hist.Hist], save_dir: str, save_prefix: str, atlas_marker: str = "Internal"
) -> None:
    figs = []

    for k, h in hists.items():
        fig, ax = plt.subplots(figsize=(7, 5.8))

        plot_hist(h, ax=ax, histtype="step", lw=2)

        ax.set_xlabel(k.capitalize())
        ax.set_ylabel("Events")
        ax.set_yscale("log")

        ax.set_ylim(1, None)

        if atlas_marker is not None:
            atlas_label(ax, loc=0, llabel=atlas_marker)

        figs.append(fig)
        plt.close(fig)

    with PdfPages(f"{save_dir}/{save_prefix}density_ratio.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
