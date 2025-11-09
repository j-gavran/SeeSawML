import argparse
import logging
import os

import hist
import matplotlib.pyplot as plt
from f9columnar.ml.dataloader_helpers import ColumnSelection
from f9columnar.ml.hdf5_dataloader import WeightedDatasetBatch, get_ml_hdf5_dataloader
from matplotlib.backends.backend_pdf import PdfPages
from plothist import plot_hist
from tqdm import tqdm

from seesaw.utils.features import PhysicsFeature, get_feature
from seesaw.utils.helpers import setup_analysis_dirs
from seesaw.utils.hydra_initalize import get_hydra_config
from seesaw.utils.loggers import setup_logger
from seesaw.utils.plots_utils import atlas_label


def make_hists(
    selection: ColumnSelection,
    nbins: int,
    scale: bool,
    logx: bool = False,
    feature_dct: dict[str, PhysicsFeature] | None = None,
) -> tuple[dict[str, list[hist.Hist]], int]:
    hs: dict[str, list[hist.Hist]] = {}

    total_hists = 0

    for k in selection.keys():
        for column_name in selection[k].offset_used_columns:
            if feature_dct is not None and column_name in feature_dct:
                feature = feature_dct[column_name]
            else:
                feature = get_feature(column_name, nbins, logx=logx)

            if scale:
                bins = feature.binning(scaled=True)
            else:
                bins = feature.binning(scaled=False)

            hs[column_name] = []
            for _ in range(selection[k].n_objects):
                h = hist.Hist(hist.axis.Variable(bins, name=str(feature), overflow=True), storage=hist.storage.Weight())
                hs[column_name].append(h)
                total_hists += 1

    return hs, total_hists


def fill_hists(
    ds_batch: WeightedDatasetBatch,
    selection: ColumnSelection,
    hs: dict[str, list[hist.Hist]],
    weighted: bool,
) -> None:
    if "events" not in ds_batch.keys():
        w = None
    else:
        w = ds_batch["events"].w

    for k in selection.keys():
        n_objects, pad_value = selection[k].n_objects, selection[k].pad_value

        for c, column_name in enumerate(selection[k].offset_used_columns):
            for i in range(selection[k].n_objects):
                w_pad = None
                if n_objects == 1:
                    x = ds_batch[k].X[:, c].numpy()
                else:
                    x = ds_batch[k].X[:, i, c].numpy()
                    if pad_value is not None:
                        padding_mask = x != pad_value
                        x = x[padding_mask]
                        w_pad = w[padding_mask] if w is not None else None
                    else:
                        w_pad = w

                if not weighted or w is None:
                    hs[column_name][i].fill(x)
                else:
                    hs[column_name][i].fill(x, weight=w if w_pad is None else w_pad)


def plot_piles_hists(hs: dict[str, list[hist.Hist]], config_name: str, scale: bool, weighted: bool) -> None:
    figs, axs = [], []

    for h_name, h_lst in tqdm(hs.items(), total=len(hs), desc="Plotting histograms", leave=False):
        for i, h in enumerate(h_lst):
            for logy in [True, False]:
                fig, ax = plt.subplots()

                plot_hist(h, ax=ax, histtype="step", linewidth=1.2, linestyle="-")

                if len(h_lst) > 1:
                    ax.set_xlabel(f"{h_name} (particle {i + 1})")
                else:
                    ax.set_xlabel(h_name)

                ax.set_ylabel("Events")

                if logy:
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(0, None)

                atlas_label(ax, loc=1, fontsize=10)

                figs.append(fig)
                axs.append(ax)
                plt.close(fig)

    save_dir = os.environ["ANALYSIS_ML_RESULTS_DIR"]

    if scale:
        save_name = f"{config_name}_scaled_piles"
    else:
        save_name = f"{config_name}_piles"

    if weighted:
        save_name += "_weighted"
    else:
        save_name += "_unweighted"

    with PdfPages(os.path.join(save_dir, f"{save_name}.pdf")) as pdf:
        for fig in tqdm(figs, total=len(figs), desc="Saving figures to PDF", leave=False):
            pdf.savefig(fig, bbox_inches="tight")


def run_plot_piles(
    files: str | list[str],
    column_names: list[str],
    stage_split_piles: dict[str, int],
    config_name: str,
    dataset_kwargs: dict | None = None,
    dataloader_kwargs: dict | None = None,
    nbins: int = 100,
    scale: bool = False,
    weighted: bool = False,
) -> None:
    logging.info("[green]Initializing dataloaders...[/green]")
    hdf5_dataloader, selection, num_entries = get_ml_hdf5_dataloader(
        name="pilesPlotter",
        files=files,
        column_names=column_names,
        stage_split_piles=stage_split_piles,  # type: ignore
        stage="plot",
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    hs, _ = make_hists(selection, nbins, scale)

    p_bar = tqdm(desc="Iterating events", total=num_entries, leave=False)

    for batch in hdf5_dataloader:
        ds_batch, _ = batch

        fill_hists(ds_batch, selection, hs, weighted)

        n = ds_batch.dim
        p_bar.update(n)

    p_bar.close()

    plot_piles_hists(hs, config_name, scale, weighted)


def main():
    parser = argparse.ArgumentParser(description="Plot hdf5 datasets saved in piles.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Configuration to use.",
    )
    parser.add_argument(
        "-p",
        "--piles",
        required=True,
        type=int,
        help="Number of piles to plot.",
    )
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
        "-w",
        "--weighted",
        action="store_true",
        help="Use weighted histograms.",
    )

    args = parser.parse_args()

    config = get_hydra_config(os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], args.config), "training_config")
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    stage_split_piles = {"plot": args.piles}

    dataset_kwargs = dict(config.dataset_config.dataset_kwargs)

    if args.scale:
        logging.info("Using feature scaling.")
        dataset_kwargs["scaler_type"] = config.dataset_config.feature_scaling.scaler_type
        dataset_kwargs["scaler_path"] = config.dataset_config.feature_scaling.save_path
        dataset_kwargs["scalers_extra_hash"] = str(config.dataset_config.files)

    if args.weighted:
        logging.info("Using weighted histograms.")
    else:
        logging.info("Using unweighted histograms.")

    run_plot_piles(
        config.dataset_config.files,
        config.dataset_config.features,
        stage_split_piles,
        args.config,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dict(config.dataset_config.dataloader_kwargs),
        nbins=args.nbins,
        scale=args.scale,
        weighted=args.weighted,
    )


if __name__ == "__main__":
    main()
