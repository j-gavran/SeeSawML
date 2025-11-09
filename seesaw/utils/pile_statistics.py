import argparse
import glob
import logging
import os
import re
from typing import Any

import numpy as np
from f9columnar.ml.dataloader_helpers import get_hdf5_metadata, get_hdf5_shapes
from f9columnar.ml.hdf5_dataloader import get_ml_hdf5_dataloader

from seesaw.utils.loggers import get_batch_progress, setup_logger


def get_detailed_statistics(
    piles_path: str, feature: str = "ptl1", num_workers: int = -1
) -> dict[str, dict[str, tuple[int, ...]]]:
    piles_path_wildcard = os.path.join(piles_path, "*")

    dataloader_kwargs: dict[str, Any] = {}
    dataloader_kwargs["batch_size"] = None
    dataloader_kwargs["num_workers"] = num_workers

    dataset_kwargs: dict[str, Any] = {}
    dataset_kwargs["drop_last"] = False

    metadata = get_hdf5_metadata(piles_path_wildcard, resolve_path=True)
    labels = metadata.get("labels", None)

    logging.info("[green]Initializing dataloaders...[/green]")
    hdf5_dataloader, selection, _ = get_ml_hdf5_dataloader(
        "pile_statistics",
        piles_path_wildcard,
        [feature, "label_type"],
        stage_split_piles=None,
        stage=None,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )
    dataset_keys = list(selection.keys())

    progress = get_batch_progress()
    progress.start()
    bar = progress.add_task("Processing batches", total=None)

    detail_stats_dct: dict[str, dict[int, int]] = {}

    total_events = 0

    for batch in hdf5_dataloader:
        ds_batch, reports = batch

        p_name = os.path.basename(reports["file"]).split(".")[0]
        y = ds_batch["events"].y.numpy()
        un_y = np.unique(y, return_counts=True)

        if p_name not in detail_stats_dct:
            detail_stats_dct[p_name] = {}

        for k, v in zip(un_y[0], un_y[1]):
            k = int(k)
            if k not in detail_stats_dct[p_name]:
                detail_stats_dct[p_name][k] = 0

            detail_stats_dct[p_name][k] += int(v)

        total_events += ds_batch[dataset_keys[0]].X.shape[0]

        progress.update(bar, advance=1)

    progress.stop()
    logging.info(f"Processed a total of {total_events} events.")

    inv_labels = {v: k for k, v in labels.items()}

    labels_detailed_stats_dct: dict[str, dict[str, tuple[int, ...]]] = {}
    for p_name, stats in detail_stats_dct.items():
        labels_detailed_stats_dct[p_name] = {}
        for k, v in stats.items():
            label_name = inv_labels[k]
            labels_detailed_stats_dct[p_name][label_name] = v

    return labels_detailed_stats_dct


def main(piles_path: str, detail: bool = False, feature: str = "ptl1", num_workers: int = -1) -> None:
    hdf5_files = glob.glob(os.path.join(piles_path, "*.hdf5"))

    if detail:
        detailed_stats_dct = get_detailed_statistics(piles_path, feature=feature, num_workers=num_workers)

    files_stats: dict[int, dict[str, tuple[int, ...]]] = {}

    for hdf5_file in hdf5_files:
        match = re.search(r"\d+", os.path.basename(hdf5_file))
        if match:
            p_number = int(match.group())
        else:
            raise ValueError(f"Could not extract pile number from filename: {hdf5_file}")

        files_stats[p_number] = {}

        shapes_dct = get_hdf5_shapes(hdf5_file)
        for key, shape in shapes_dct.items():
            files_stats[p_number][key] = shape

    files_stats = dict(sorted(files_stats.items()))

    file_stats_str = ""

    file_stats_str += f"Found {len(hdf5_files)} HDF5 piles in {piles_path}:\n"

    total_counts: dict[str, int] = {}
    lst_counts_dct: dict[str, list[int]] = {}
    for file_idx, stats in files_stats.items():
        file_stats_str += f"Pile {file_idx}:\n"
        for key, shape in stats.items():
            file_stats_str += f"  {key}: {shape}\n"
            if key not in total_counts:
                total_counts[key] = 0

            if key not in lst_counts_dct:
                lst_counts_dct[key] = []

            total_counts[key] += shape[0]
            lst_counts_dct[key].append(shape[0])

        if detail:
            file_stats_str += "  Detailed statistics:\n"
            stats = detailed_stats_dct[f"p{file_idx}"]
            for label_name, count in stats.items():
                file_stats_str += f"    {label_name}: {count}\n"

    file_stats_str += "Total counts across all piles:\n"
    for key, total in total_counts.items():
        file_stats_str += f"  {key}: {total}\n"

    file_stats_str += "Mean and std of counts across all piles:\n"
    for key, lst_counts in lst_counts_dct.items():
        mean = np.mean(lst_counts)
        std = np.std(lst_counts)
        file_stats_str += f"  {key}: {mean:.2f} +/- {std:.2f}\n"

    with open(os.path.join(piles_path, "piles_statistics.txt"), "w") as f:
        f.write(file_stats_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics of HDF5 piles.")
    parser.add_argument(
        "--piles_path",
        type=str,
        required=False,
        help="Path to the directory containing HDF5 piles.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Whether to print detailed statistics for each pile.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--representative_feature",
        type=str,
        default="ptl1",
        help="Representative feature for detailed statistics.",
    )
    args = parser.parse_args()

    if args.piles_path is not None:
        piles_path = args.piles_path
    else:
        piles_path = os.environ["ANALYSIS_ML_DATA_DIR"]

    setup_logger()

    main(piles_path, detail=args.detail, feature=args.representative_feature, num_workers=args.num_workers)
