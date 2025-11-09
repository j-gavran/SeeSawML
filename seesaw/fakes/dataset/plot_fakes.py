import argparse
import logging
import os

import hist
import matplotlib.pyplot as plt
import numpy as np
from f9columnar.ml.hdf5_dataloader import get_ml_hdf5_dataloader
from matplotlib.backends.backend_pdf import PdfPages
from plothist import create_comparison_figure, plot_data_model_comparison
from tqdm import tqdm

from seesaw.fakes.utils import handle_fakes_dataset
from seesaw.utils.features import get_feature
from seesaw.utils.helpers import setup_analysis_dirs
from seesaw.utils.hydra_initalize import get_hydra_config
from seesaw.utils.loggers import setup_logger
from seesaw.utils.plot_piles import fill_hists, make_hists
from seesaw.utils.plots_utils import atlas_label


def plot_fakes_hists(
    hs: list[dict[str, list[hist.Hist]]],
    scale: bool,
    remove_comparison: bool = False,
    atlas_marker: str = "Internal",
) -> None:
    hs_dct = {"loose": {"data": hs[0], "mc": hs[2]}, "tight": {"data": hs[1], "mc": hs[3]}}

    figs = []
    for loose_tight_name, data_mc_dct in hs_dct.items():
        data_hs_dct, mc_hs_dct = data_mc_dct["data"], data_mc_dct["mc"]

        h_names = list(data_hs_dct.keys())

        for h_name in tqdm(h_names, desc=f"Plotting {loose_tight_name} histograms", total=len(h_names), leave=False):
            h_data, h_mc = data_hs_dct[h_name][0], mc_hs_dct[h_name][0]
            xlabel = f"{loose_tight_name.title()} {str(get_feature(h_name))}"

            for logy in [True, False]:
                fig, (ax_main, ax_comparison) = create_comparison_figure(
                    figsize=(6, 6.5) if remove_comparison else (6, 5)
                )

                plot_data_model_comparison(
                    data_hist=h_data,
                    unstacked_components=[h_mc],
                    unstacked_labels=["MC"],
                    unstacked_colors=["C0"],
                    xlabel=xlabel,
                    ylabel="Events",
                    model_sum_kwargs={"show": False},
                    comparison_ylim=[0.5, 1.5],
                    data_uncertainty_type="symmetrical",
                    comparison="split_ratio",
                    model_uncertainty=True,
                    fig=fig,
                    ax_main=ax_main,
                    ax_comparison=ax_comparison,
                )

                if remove_comparison:
                    fig.delaxes(ax_comparison)
                    ax_comparison = None
                    ax_main.set_xlabel(xlabel)

                if not scale and ("pt" in h_name or "met" in h_name):
                    ax_main.set_xscale("log")
                    if ax_comparison is not None:
                        ax_comparison.set_xscale("log")

                if logy:
                    ax_main.set_yscale("log")
                    ax_main.set_ylim(0.1, None)
                else:
                    ax_main.set_ylim(0.0, None)

                for line in ax_main.lines:
                    line.set_linewidth(2)

                if atlas_marker != "none":
                    atlas_label(ax_main, loc=1, fontsize=10)

                figs.append(fig)
                plt.close(fig)

    save_dir = os.environ["ANALYSIS_ML_RESULTS_DIR"]

    if scale:
        save_name = "scaled_loose_tight_fakes"
    else:
        save_name = "fakes_loose_tight"

    with PdfPages(os.path.join(save_dir, f"{save_name}.pdf")) as pdf:
        for fig in tqdm(figs, total=len(figs), desc="Saving figures to PDF", leave=False):
            pdf.savefig(fig, bbox_inches="tight")


def run_plot_fakes(
    files: str | list[str],
    column_names: list[str],
    dataset_kwargs: dict | None = None,
    dataloader_kwargs: dict | None = None,
    nbins: int = 100,
    scale: bool = False,
    remove_comparison: bool = False,
    atlas_marker: str = "Internal",
) -> None:
    if dataset_kwargs is None:
        dataset_kwargs = {}

    dataset_kwargs["setup_func"] = handle_fakes_dataset

    dataset_kwargs.update({"use_loose": True, "use_tight": True, "use_data": True, "use_mc": True})

    logging.info("[green]Initializing dataloaders...[/green]")
    hdf5_dataloader, selection, num_entries = get_ml_hdf5_dataloader(
        name="fakesPlotter",
        files=files,
        column_names=column_names,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    options = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])  # data/loose, data/tight, mc/loose, mc/tight

    hs_options: list[dict[str, list[hist.Hist]]] = []
    for _ in range(len(options)):
        hs, _ = make_hists(selection, nbins, scale, logx=False if scale else True)
        hs_options.append(hs)

    p_bar = tqdm(desc="Iterating events", total=num_entries, leave=False)

    for batch in hdf5_dataloader:
        ds_batch, _ = batch

        for i, option in enumerate(options):
            mask = (ds_batch["events"].y == option[0]) & (ds_batch["events"].y_aux == option[1])
            ds_masked_batch = ds_batch.mask(mask)

            fill_hists(ds_masked_batch, selection, hs_options[i], weighted=True)

        n = ds_batch.dim
        p_bar.update(n)

    p_bar.close()

    plot_fakes_hists(hs_options, scale, remove_comparison=remove_comparison, atlas_marker=atlas_marker)


def main():
    parser = argparse.ArgumentParser(description="Plot hdf5 fakes dataset.")

    parser.add_argument(
        "-n",
        "--nbins",
        type=int,
        default=100,
        help="Number of bins for histograms.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        action="store_true",
        help="Scale the features.",
    )
    parser.add_argument(
        "--remove-comparison",
        action="store_true",
        help="Remove the comparison subplot from the plots.",
    )
    parser.add_argument(
        "-a",
        "--atlas",
        type=str,
        default="Internal",
        help="Add ATLAS label to the plots.",
    )

    args = parser.parse_args()

    config = get_hydra_config(os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "fakes"), "training_config")
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    dataset_kwargs = dict(config.dataset_config.dataset_kwargs)
    dataloader_kwargs = dict(config.dataset_config.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null for plotting.")

    if args.scale:
        logging.info("Using feature scaling.")
        dataset_kwargs["scaler_type"] = config.dataset_config.feature_scaling.scaler_type
        dataset_kwargs["scaler_path"] = config.dataset_config.feature_scaling.save_path

    run_plot_fakes(
        config.dataset_config.files,
        config.dataset_config.features,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        nbins=args.nbins,
        scale=args.scale,
        remove_comparison=args.remove_comparison,
        atlas_marker=args.atlas,
    )


if __name__ == "__main__":
    main()
