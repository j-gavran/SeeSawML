import copy
import logging
import os
from typing import Any

import hist
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from f9columnar.ml.dataloader_helpers import ColumnSelection, column_selection_from_dict, get_hdf5_metadata
from f9columnar.ml.hdf5_dataloader import (
    MLHdf5Iterator,
    StackedDatasets,
    WeightedBatch,
    WeightedDatasetBatch,
    get_ml_hdf5_dataloader,
    remap_labels_lookup,
)
from f9columnar.utils.loggers import get_progress
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import DictConfig, ListConfig
from plothist import get_color_palette, plot_data_model_comparison, plot_model

from seesaw.signal.training.sig_bkg_trainer import load_sig_bkg_model
from seesaw.signal.utils import get_classifier_labels, torch_multiclass_discriminant
from seesaw.utils.features import PhysicsFeature
from seesaw.utils.helpers import setup_analysis_dirs
from seesaw.utils.labels import get_label
from seesaw.utils.loggers import get_batch_progress, setup_logger
from seesaw.utils.plot_piles import fill_hists, make_hists
from seesaw.utils.plots_utils import atlas_label


def plot_signal_setup_func(stacked_datasets: StackedDatasets, ml_iterator: MLHdf5Iterator) -> WeightedDatasetBatch:
    weighted_datase_batch = WeightedDatasetBatch()

    X: np.ndarray = stacked_datasets["events"].X.astype(np.float32)
    y: np.ndarray = stacked_datasets["events"].get_extra("label_type")  # type: ignore[assignment]
    w: np.ndarray = stacked_datasets["events"].get_extra("weights")  # type: ignore[assignment]

    remap_labels: dict[int, int] | None = ml_iterator.dataset_kwargs.get("remap_labels", None)
    mask_unmapped: np.ndarray | None = None

    if remap_labels is not None:
        max_label = ml_iterator.dataset_kwargs["max_label"]
        y, mask_unmapped = remap_labels_lookup(y, max_label, remap_labels)
        X, w = X[mask_unmapped], w[mask_unmapped]

    weighted_datase_batch["events"] = WeightedBatch(X, y, w, None)

    for ds_name, ds in stacked_datasets.items():
        if ds_name == "events":
            continue

        X = ds.X.astype(np.float32)
        if mask_unmapped is not None:
            X = X[mask_unmapped]

        weighted_datase_batch[ds_name] = WeightedBatch(X, None, None, None)

    return weighted_datase_batch


def plot_signal_hists(
    hs: dict[str, dict[str, list[hist.Hist]]],
    labels: dict[str, int],
    scale: bool,
    weighted: bool,
    signal_name: str | None,
    is_closure: bool = False,
    blinded: bool = True,
) -> None:
    figs, axs = [], []

    if signal_name is None:
        stacked_labels = list(labels.keys())
        unstacked_labels = []
    else:
        stacked_labels = [label for label in labels.keys() if label != signal_name]
        unstacked_labels = [signal_name]

    if not blinded:
        stacked_labels = [label for label in stacked_labels if label != "data"]

    if len(stacked_labels) == 1:
        colors = [get_color_palette("YlGnBu_r", 10)[4]]
    else:
        colors = get_color_palette("YlGnBu_r", len(stacked_labels) + 1)[1:]

    # variable name: label name: list of histograms
    variable_label_hs: dict[str, dict[str, list[hist.Hist]]] = {}
    for label_name, variable_dct in hs.items():
        for h_name, h_lst in variable_dct.items():
            if h_name not in variable_label_hs:
                variable_label_hs[h_name] = {}

            variable_label_hs[h_name][label_name] = h_lst

    progress = get_progress()
    progress.start()
    bar = progress.add_task("Plotting histograms", total=len(variable_label_hs))

    for h_name, labels_dct in variable_label_hs.items():
        stacked_components_obj_lst, unstacked_components_obj_lst, data = [], [], None
        for label_name, h_lst in labels_dct.items():
            if not blinded and label_name == "data":
                data = h_lst
                continue

            stacked_lst, unstacked_lst = [], []
            for h in h_lst:
                if signal_name is not None and label_name == signal_name:
                    unstacked_lst.append(h)
                else:
                    stacked_lst.append(h)

            if len(stacked_lst) != 0:
                stacked_components_obj_lst.append(stacked_lst)

            if len(unstacked_lst) != 0:
                unstacked_components_obj_lst.append(unstacked_lst)

        n_objects = len(h_lst)

        for i in range(n_objects):
            if signal_name is not None:
                unstacked_components = [h[i] for h in unstacked_components_obj_lst]
            else:
                unstacked_components = []

            stacked_components = [h[i] for h in stacked_components_obj_lst]

            if data is not None:
                data_i = data[i]

            xlabel_name = str(stacked_components[0].axes[0].name)

            if n_objects > 1:
                xlabel = f"{xlabel_name} (particle {i + 1})"
            else:
                xlabel = xlabel_name

            idx = np.argsort([h.values().sum() for h in stacked_components])

            sorted_stacked_components = [stacked_components[j] for j in idx]
            sorted_stacked_labels = [stacked_labels[j] for j in idx]

            stacked_labels_plot = [get_label(l).latex_name for l in sorted_stacked_labels]
            unstacked_labels_plot = [get_label(l).latex_name for l in unstacked_labels]

            for logy in [False, True]:
                if data is None:
                    fig, ax = plot_model(
                        stacked_components=sorted_stacked_components,
                        stacked_labels=stacked_labels_plot,
                        stacked_colors=colors,
                        unstacked_components=unstacked_components,
                        unstacked_labels=unstacked_labels_plot,
                        unstacked_colors=["red"],
                        unstacked_kwargs_list=[{"linestyle": "dotted", "linewidth": 1.2}],
                        xlabel=xlabel,
                        ylabel="Events",
                        model_sum_kwargs={"show": False},
                        model_uncertainty_label="Stat. unc.",
                    )
                else:
                    fig, ax, _ = plot_data_model_comparison(
                        data_hist=data_i,
                        stacked_components=sorted_stacked_components,
                        stacked_labels=stacked_labels_plot,
                        stacked_colors=colors,
                        unstacked_components=unstacked_components,
                        unstacked_labels=unstacked_labels_plot,
                        unstacked_colors=["red"],
                        unstacked_kwargs_list=[{"linestyle": "dotted", "linewidth": 1.2}],
                        xlabel=xlabel,
                        ylabel="Events",
                        model_sum_kwargs={"show": False},
                        model_uncertainty_label="Stat. unc.",
                        data_uncertainty_type="symmetrical",
                    )

                if logy:
                    ax.set_yscale("log")
                    if weighted:
                        ax.set_ylim(1e-3, None)
                    else:
                        ax.set_ylim(1, None)
                else:
                    ax.set_ylim(0, None)

                ax.legend(loc="upper right", fontsize=7, ncol=2)

                atlas_label(ax, loc=1, fontsize=10)

                figs.append(fig)
                axs.append(ax)
                plt.close(fig)

        progress.update(bar, advance=1)

    progress.stop()

    save_dir = f"{os.environ['ANALYSIS_ML_RESULTS_DIR']}/signal_plots"
    os.makedirs(save_dir, exist_ok=True)

    if scale:
        save_name = "scaled_signal"
    else:
        save_name = "signal"

    if weighted:
        save_name += "_weighted"
    else:
        save_name += "_unweighted"

    if not blinded:
        save_name += "_unblinded"

    if is_closure:
        save_name += "_closure"

    progress = get_progress()
    progress.start()
    bar = progress.add_task("Saving figures to PDF", total=len(figs))

    with PdfPages(os.path.join(save_dir, f"{save_name}.pdf")) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
            progress.update(bar, advance=1)

    progress.stop()


def dump_hists_to_csv(
    hs: dict[str, dict[str, list[hist.Hist]]],
    labels: dict[str, int],
    save_dir: str,
    save_name: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # variable -> label -> list[Hist]
    variable_label_hs: dict[str, dict[str, list[hist.Hist]]] = {}
    for label_name, variable_dct in hs.items():
        for h_name, h_lst in variable_dct.items():
            if h_name not in variable_label_hs:
                variable_label_hs[h_name] = {}
            variable_label_hs[h_name][label_name] = h_lst

    for h_name, labels_dct in variable_label_hs.items():
        rows: list[dict[str, float | int | str]] = []

        # Assume consistent binning across labels for this variable
        any_label = next(iter(labels_dct))
        n_objects = len(labels_dct[any_label])

        for obj_idx in range(n_objects):
            # Derive bin edges from any label's histogram for this object index
            any_hist = labels_dct[any_label][obj_idx]
            edges = any_hist.axes[0].edges

            for label_name, h_list in labels_dct.items():
                h = h_list[obj_idx]
                values = h.values()
                variances = h.variances() if hasattr(h, "variances") else None

                for b in range(len(edges) - 1):
                    val = float(values[b])
                    var = float(variances[b]) if variances is not None else 0.0
                    rows.append(
                        {
                            "variable": h_name,
                            "object_index": obj_idx,
                            "bin_left": float(edges[b]),
                            "bin_right": float(edges[b + 1]),
                            "label": label_name,
                            "value": val,
                            "variance": var,
                        }
                    )

        # Write CSV using pandas
        out_path = os.path.join(save_dir, f"{save_name}_{h_name}.csv")
        pd.DataFrame.from_records(rows).to_csv(out_path, index=False)


def dump_score_values_to_csv(values_dct: dict[str, dict[str, list[float]]], save_dir: str, save_name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Collect variable names present across labels (e.g., mva_score, mva_disc)
    variable_names: set[str] = set()
    for label_values in values_dct.values():
        variable_names.update(label_values.keys())

    for var_name in sorted(variable_names):
        rows: list[dict[str, float | str]] = []
        for label_name, label_values in values_dct.items():
            if var_name not in label_values:
                continue
            for v in label_values[var_name]:
                rows.append({"label": label_name, "value": float(v)})

        out_path = os.path.join(save_dir, f"{save_name}_{var_name}_values.csv")
        pd.DataFrame.from_records(rows).to_csv(out_path, index=False)


def rescale_dataset_batch(
    ds_masked_batch: WeightedDatasetBatch,
    used_selection: ColumnSelection,
    reports: dict[str, Any],
    has_model: bool = False,
    inverse: bool = True,
) -> WeightedDatasetBatch:
    for dataset_name in ds_masked_batch.keys():
        numer_scalers = reports["numer_feature_scaler_dct"][dataset_name]
        categ_scalers = reports["categ_feature_scaler_dct"][dataset_name]

        offset_numer_columns_idx = used_selection[dataset_name].offset_numer_columns_idx
        offset_categ_columns_idx = used_selection[dataset_name].offset_categ_columns_idx

        n_objects = used_selection[dataset_name].n_objects

        if has_model:
            if n_objects != 1:
                raise NotImplementedError("Model rescaling not implemented for multi-object datasets.")

            y_model = ds_masked_batch[dataset_name].X[:, offset_numer_columns_idx[-1]].numpy()
            offset_numer_columns_idx = offset_numer_columns_idx[:-1]

        if n_objects == 1:
            X_numer = ds_masked_batch[dataset_name].X[:, offset_numer_columns_idx].numpy()
            X_categ = ds_masked_batch[dataset_name].X[:, offset_categ_columns_idx].numpy()

            if inverse:
                X_numer_scaled = numer_scalers[0].inverse_transform(X_numer)
                X_categ_scaled = categ_scalers[0].inverse_transform(X_categ)
            else:
                X_numer_scaled = numer_scalers[0].transform(X_numer)
                X_categ_scaled = categ_scalers[0].transform(X_categ)

            if has_model:
                X_scaled = np.concatenate([X_numer_scaled, y_model[:, None], X_categ_scaled], axis=1)
            else:
                X_scaled = np.concatenate([X_numer_scaled, X_categ_scaled], axis=1)
        else:
            X_numer = ds_masked_batch[dataset_name].X[:, :, offset_numer_columns_idx].numpy()
            X_categ = ds_masked_batch[dataset_name].X[:, :, offset_categ_columns_idx].numpy()

            X_numer_scaled = np.empty_like(X_numer)
            X_categ_scaled = np.empty_like(X_categ)

            for i in range(n_objects):
                if inverse:
                    X_numer_scaled[:, i, :] = numer_scalers[i].inverse_transform(X_numer[:, i, :])
                    X_categ_scaled[:, i, :] = categ_scalers[i].inverse_transform(X_categ[:, i, :])
                else:
                    X_numer_scaled[:, i, :] = numer_scalers[i].transform(X_numer[:, i, :])
                    X_categ_scaled[:, i, :] = categ_scalers[i].transform(X_categ[:, i, :])

            X_scaled = np.concatenate([X_numer_scaled, X_categ_scaled], axis=2)

        ds_masked_batch[dataset_name] = WeightedBatch(
            X_scaled,
            ds_masked_batch[dataset_name].y,
            ds_masked_batch[dataset_name].w,
            ds_masked_batch[dataset_name].y_aux,
        )

    return ds_masked_batch


def run_plot_signal(
    files: str | list[str],
    column_names: list[str],
    dataset_kwargs: dict | None = None,
    dataloader_kwargs: dict | None = None,
    nbins: int = 100,
    scale: bool = False,
    weighted: bool = False,
    signal_name: str | None = None,
    rescale: bool = False,
    model: torch.nn.Module | None = None,
    model_cut: tuple[float, float] = (0.5, 1.0),
    ml_mass: list[float] | None = None,
    use_discriminant: bool = False,
    classes: ListConfig | None = None,
    force_is_multiclass: bool = False,
    plot_only_ml: bool = False,
    plot_both_ml_and_disc: bool = False,
    dump_csv: bool = False,
    csv_dir: str | None = None,
    custom_groups: dict[str, list[str]] | None = None,
) -> None:
    if dataset_kwargs is None:
        dataset_kwargs = {}

    metadata = get_hdf5_metadata(files, resolve_path=True)
    labels = metadata.get("labels", None)

    if labels is None:
        raise ValueError("No labels found in the metadata of the provided files!")

    dataset_kwargs["max_label"] = max(labels.values())

    if classes is not None:
        labels, remap_labels, _ = get_classifier_labels(classes, labels)
    else:
        remap_labels = None

    if signal_name is not None and signal_name not in labels:
        raise ValueError(f"Signal name '{signal_name}' not found in the labels: {list(labels.keys())}.")

    dataset_kwargs["remap_labels"] = remap_labels

    logging.info(f"Signal: {signal_name}.")

    if signal_name is None:
        logging.info(f"Background: {list(labels.keys())}.")
    else:
        logging.info(f"Background: {list(set(labels.keys()) - {signal_name})}.")

    if "data" in labels:
        blinded = False
        logging.warning("[yellow]Using unblinded dataset!")
    else:
        blinded = True

    if model is not None:
        if classes is None:
            is_multiclass = False
            logging.info("Running closure without classes (assuming binary classification).")
        elif len(labels) == 2:
            logging.info("Running closure for binary classification.")
            is_multiclass = False
        elif len(labels) > 2:
            logging.info("Running closure for multiclass classification.")
            is_multiclass = True

            signal_idx = labels.get(signal_name, None)
            if signal_idx is None:
                raise ValueError(f"Signal name '{signal_name}' not found in the labels: {list(labels.keys())}.")
            else:
                logging.info(f"Multiclass signal index: {signal_idx}.")
        else:
            raise ValueError(f"Expected at least two labels, got {len(labels)}: {list(labels.keys())}.")

        if force_is_multiclass:
            is_multiclass = True
            logging.info("Forcing multiclass classification.")
        else:
            is_multiclass = False
            signal_idx = None
            logging.info("Forcing binary classification (no signal index).")

    dataset_kwargs["setup_func"] = plot_signal_setup_func

    logging.info("[green]Initializing dataloaders...[/green]")
    hdf5_dataloader, selection, _ = get_ml_hdf5_dataloader(
        name="signalPlotter",
        files=files,
        column_names=column_names,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
    )

    if model is not None:
        if len(selection) != 1:
            raise NotImplementedError(f"Expected only one dataset in the selection, got {len(selection)}.")

        if "events" not in selection:
            raise NotImplementedError("The selection must contain only the 'events' dataset with features and labels.")

        base_selection_dict = copy.deepcopy(selection["events"]).to_dict()

        # Build selection dict(s) for ML-only or appended mode
        ml_columns: list[str] = ["mva_score"]
        if plot_both_ml_and_disc:
            ml_columns.append("mva_disc")

        if plot_only_ml:
            # Only plot ML columns
            dct_selection = {
                "all_columns": ml_columns.copy(),
                "used_columns": ml_columns.copy(),
                "extra_columns": [],
                "numer_columns": ml_columns.copy(),
                "categ_columns": [],
                "numer_columns_idx": list(range(len(ml_columns))),
                "categ_columns_idx": [],
                "shape": (base_selection_dict["shape"][0], len(ml_columns)),
                # Optional keys consumed with defaults inside column_selection_from_dict
                "pad_value": base_selection_dict.get("pad_value", None),
                "labels": base_selection_dict.get("labels", None),
            }
            model_selection = column_selection_from_dict({"events": dct_selection})
        else:
            # Append ML score (and optionally disc) to the existing selection
            dct_selection = copy.deepcopy(base_selection_dict)
            for new_col in ml_columns:
                dct_selection["all_columns"].append(new_col)
                dct_selection["used_columns"].append(new_col)
                dct_selection["numer_columns"].append(new_col)
                dct_selection["shape"] = (dct_selection["shape"][0], dct_selection["shape"][1] + 1)
                numer_columns_lst = list(dct_selection["numer_columns_idx"])
                dct_selection["numer_columns_idx"] = numer_columns_lst + [len(dct_selection["used_columns"]) - 1]
            model_selection = column_selection_from_dict({"events": dct_selection})

    used_selection = model_selection if model else selection

    if ml_mass is not None:
        columns = selection["events"].offset_used_columns
        ml_mass_indices = []
        for i, c in enumerate(columns):
            if "mlmass" in c:
                ml_mass_indices.append(i)

        if len(ml_mass_indices) != len(ml_mass):
            raise ValueError("Number of ML mass values does not match the number of ML mass indices!")

        logging.info(f"Setting ML mass values to {ml_mass} at indices {ml_mass_indices}.")
    else:
        ml_mass_indices = []

    feature_dct = {
        "mva_score": PhysicsFeature(
            "mva_score",
            nbins,
            x_range=model_cut,
            x_range_scaled=model_cut,
            latex_name=f"MVA score cut in range {model_cut}",
            logx=False,
        )
    }
    if plot_both_ml_and_disc:
        feature_dct["mva_disc"] = PhysicsFeature(
            "mva_disc",
            nbins,
            x_range=model_cut,
            x_range_scaled=model_cut,
            latex_name=f"Multiclass discriminant in range {model_cut}",
            logx=False,
        )

    # label name: variable name: list of histograms
    hs_labels: dict[str, dict[str, list[hist.Hist]]] = {}
    for label_name in labels.keys():
        hs, _ = make_hists(used_selection, nbins, scale, feature_dct=feature_dct)
        hs_labels[label_name] = hs

    counts = {label_name: 0 for label_name in labels.keys()}

    # Collect per-event values used for histograms (e.g., mva_score, optionally mva_disc)
    score_values: dict[str, dict[str, list[float]]] = {label_name: {} for label_name in labels.keys()}

    if model is not None:
        device = str(model.device)

    progress = get_batch_progress()
    progress.start()
    bar = progress.add_task("Processing batches", total=None)

    # Accumulate predictions and true labels (one-hot) for group plotting
    all_y_pred: list[np.ndarray] = []
    all_y_true_onehot: list[np.ndarray] = []

    for batch in hdf5_dataloader:
        ds_batch, reports = batch

        # If model is available and this is a multiclass problem, accumulate predictions/labels for group plots
        if model is not None and len(labels) > 2:
            X_all = ds_batch["events"].X.to(device)
            with torch.no_grad():
                logits_all = model(X_all)
                probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()

            y_idx = ds_batch["events"].y
            if y_idx is not None:
                y_idx_np = y_idx.cpu().numpy().astype(int)
                num_classes = len(labels)
                y_onehot = np.eye(num_classes, dtype=np.float32)[y_idx_np]
                all_y_pred.append(probs_all)
                all_y_true_onehot.append(y_onehot)

        for label_name, label_value in labels.items():
            mask = ds_batch["events"].y == label_value
            ds_masked_batch = ds_batch.mask(mask)

            if model is not None:
                if ml_mass is not None:
                    ds_masked_batch = rescale_dataset_batch(ds_masked_batch, selection, reports, has_model=False)

                    X = ds_masked_batch["events"].X

                    for i, ml_mass_idx in enumerate(ml_mass_indices):
                        X[:, ml_mass_idx] = torch.tensor(ml_mass[i], device=X.device, dtype=X.dtype)

                    ds_masked_batch["events"] = WeightedBatch(
                        X,
                        ds_masked_batch["events"].y,
                        ds_masked_batch["events"].w,
                        ds_masked_batch["events"].y_aux,
                    )

                    ds_masked_batch = rescale_dataset_batch(
                        ds_masked_batch, selection, reports, has_model=False, inverse=False
                    )

                X = ds_masked_batch["events"].X.to(device)

                with torch.no_grad():
                    if is_multiclass:
                        logits = model(X)
                        probs = torch.softmax(logits, dim=1)
                        y_score = probs[:, signal_idx].unsqueeze(1)
                        y_disc_vec = torch_multiclass_discriminant(probs)
                        y_disc = y_disc_vec[signal_idx].unsqueeze(1)
                    else:
                        logits = model(X)
                        y_score = torch.sigmoid(logits)
                        y_disc = y_score

                if plot_only_ml:
                    cols = [y_score]
                    if plot_both_ml_and_disc:
                        cols.append(y_disc)
                    X = torch.concatenate(cols, dim=1)
                else:
                    cols = [
                        X[:, selection["events"].offset_numer_columns_idx],
                        y_score,
                        X[:, selection["events"].offset_categ_columns_idx],
                    ]
                    if plot_both_ml_and_disc:
                        # Append discriminant after score
                        cols.insert(2, y_disc)
                    X = torch.concatenate(cols, dim=1)

                # For plotting-only-ML, apply cut on score. Otherwise keep previous behavior (score-based cut)
                y_model_for_cut = y_score
                y_model_cut = (model_cut[0] <= y_model_for_cut) & (y_model_for_cut <= model_cut[1])

                # Accumulate per-event values that are histogrammed
                mask_flat = y_model_cut.flatten()
                # mva_score values
                score_vals = y_score[mask_flat].flatten().cpu().numpy().tolist()
                if "mva_score" not in score_values[label_name]:
                    score_values[label_name]["mva_score"] = []
                score_values[label_name]["mva_score"].extend([float(v) for v in score_vals])
                # optionally mva_disc
                if plot_both_ml_and_disc:
                    disc_vals = y_disc[mask_flat].flatten().cpu().numpy().tolist()
                    if "mva_disc" not in score_values[label_name]:
                        score_values[label_name]["mva_disc"] = []
                    score_values[label_name]["mva_disc"].extend([float(v) for v in disc_vals])

                ds_masked_batch["events"] = WeightedBatch(
                    X.cpu().numpy(),
                    ds_masked_batch["events"].y,
                    ds_masked_batch["events"].w,
                    ds_masked_batch["events"].y_aux,
                )

                ds_masked_batch = ds_masked_batch.mask(y_model_cut.flatten().cpu().numpy())

            counts[label_name] += ds_masked_batch["events"].y.shape[0]

            # If plotting only ML outputs (no original features present), skip rescaling
            if rescale and not plot_only_ml:
                ds_masked_batch = rescale_dataset_batch(ds_masked_batch, used_selection, reports, has_model=True)

            fill_hists(ds_masked_batch, used_selection, hs_labels[label_name], weighted)

        progress.update(bar, advance=1)

    progress.stop()

    if signal_name is not None:
        labels.pop(signal_name)

    counts_num_entries = sum(counts.values())
    percent_counts = {k: float(f"{v / counts_num_entries * 100:.3e}") for k, v in counts.items()}

    logging.info("Counts per label:")
    for label, count in counts.items():
        logging.info(f"  {label}: {count}, {percent_counts[label]} %")

    plot_signal_hists(
        hs_labels,
        labels,
        scale,
        weighted,
        signal_name,
        is_closure=True if model is not None else False,
        blinded=blinded,
    )

    if dump_csv:
        save_dir = f"{os.environ['ANALYSIS_ML_RESULTS_DIR']}/signal_plots"
        save_name = "scaled_signal" if scale else "signal"
        save_name += "_weighted" if weighted else "_unweighted"
        if not blinded:
            save_name += "_unblinded"
        if model is not None:
            save_name += "_closure"

        out_dir = csv_dir if csv_dir is not None else save_dir
        # Dump per-event values used to build histograms
        dump_score_values_to_csv(score_values, out_dir, save_name)
        # Also dump histogram bin contents (existing behavior)
        dump_hists_to_csv(hs_labels, labels, out_dir, save_name)


@hydra.main(
    config_path=os.path.join(os.environ["ANALYSIS_ML_CONFIG_DIR"], "signal"),
    config_name="training_config",
    version_base=None,
)
def main(config: DictConfig):
    setup_logger(config.min_logging_level)
    setup_analysis_dirs(config, verbose=False)

    signal_plot_conf = config.plotting_config.signal_plot

    dataset_kwargs = dict(config.dataset_config.dataset_kwargs)

    scale = signal_plot_conf.scale

    if signal_plot_conf.closure and not signal_plot_conf.scale:
        logging.warning("Feature scaling is recommended for closure test. Force enabling scaling.")
        scale = True

    if signal_plot_conf.rescale:
        scale = True

    if scale:
        logging.info("Using feature scaling.")
        dataset_kwargs["scaler_type"] = config.dataset_config.feature_scaling.scaler_type
        dataset_kwargs["scaler_path"] = config.dataset_config.feature_scaling.save_path
        dataset_kwargs["scalers_extra_hash"] = str(config.dataset_config.files)

    if signal_plot_conf.rescale:
        logging.info("Using feature rescaling to the original range.")

    if signal_plot_conf.closure:
        logging.info("[green]Performing classification closure test.")
        device = "cuda" if config.experiment_config.accelerator == "gpu" else "cpu"
        model, load_checkpoint = load_sig_bkg_model(config.model_config, map_location=device)
        logging.info(f"Loaded model {os.path.basename(load_checkpoint)} on {device}.")
    else:
        model = None

    dataloader_kwargs = dict(config.dataset_config.dataloader_kwargs)

    if dataloader_kwargs.get("batch_size", None) is not None:
        logging.warning("Batch size is set, consider setting it to null for plotting.")

    closure_cut = (signal_plot_conf.cut_low, signal_plot_conf.cut_high)

    ml_mass = [float(mass) for mass in signal_plot_conf.ml_mass.split(",")] if signal_plot_conf.ml_mass else None

    run_plot_signal(
        config.dataset_config.files,
        config.dataset_config.features,
        dataset_kwargs=dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        nbins=signal_plot_conf.nbins,
        scale=False if signal_plot_conf.rescale else scale,
        weighted=signal_plot_conf.weighted,
        signal_name=signal_plot_conf.signal_name,
        rescale=signal_plot_conf.rescale,
        model=model,
        model_cut=closure_cut,
        ml_mass=ml_mass,
        use_discriminant=signal_plot_conf.disc,
        classes=config.dataset_config.get("classes", None),
        force_is_multiclass=signal_plot_conf.get("is_multiclass", False),
        plot_only_ml=signal_plot_conf.get("plot_only_ml", False),
        plot_both_ml_and_disc=signal_plot_conf.get("plot_both_ml_and_disc", False),
        dump_csv=signal_plot_conf.get("dump_csv", False),
        csv_dir=signal_plot_conf.get("csv_dir", None),
        custom_groups=config.dataset_config.get("custom_groups", None),
    )


if __name__ == "__main__":
    main()
