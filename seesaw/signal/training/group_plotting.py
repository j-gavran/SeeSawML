import os

import matplotlib.pyplot as plt
import numpy as np
from f9columnar.utils.helpers import handle_plot_exception
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import auc, roc_curve

from seesaw.signal.utils import multiclass_group_discriminant
from seesaw.utils.plots_utils import atlas_label, iqr_remove_outliers, save_plot


@handle_plot_exception
def plot_multiclass_group_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    index_groups: dict[str, list[int]],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
) -> None:
    bins = list(np.linspace(0, 1, nbins))

    figs = []
    for group_name, group_indices in index_groups.items():
        group_scores = np.sum(y_pred[:, group_indices], axis=1)

        signal_mask = np.isin(y_true, group_indices)
        signal_scores = group_scores[signal_mask]

        background_mask = ~signal_mask
        background_scores = group_scores[background_mask]

        fig, ax = plt.subplots(figsize=(7, 6.25))
        ax.hist(
            signal_scores,
            bins=bins,
            histtype="step",
            label=f"Signal: {group_name}",
            lw=2.0,
            color="C0",
            density=True,
        )
        ax.hist(
            background_scores,
            bins=bins,
            histtype="step",
            label="Background: rest",
            lw=2.0,
            color="C1",
            density=True,
        )

        if signal_scores.size:
            signal_min_max = (min(signal_scores), max(signal_scores))
        else:
            signal_min_max = (0.0, 0.0)

        ax.text(
            0.05,
            0.05,
            f"Signal: {group_name}\nMin: {signal_min_max[0]:.3f}, Max: {signal_min_max[1]:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
        )

        ax.set_xlabel(f"Group Score for {group_name}", fontsize=16)
        ax.set_ylabel("Events (normalized)", fontsize=16)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=12)
        atlas_label(ax, loc=0, fontsize=14)

        ax.set_xlim(-0.01, 1.01)

        figs.append(fig)
        plt.close(fig)

    os.makedirs(save_path, exist_ok=True)
    with PdfPages(f"{save_path}/custom_score_{save_postfix}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


@handle_plot_exception
def plot_multiclass_group_discriminant(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    index_groups: dict[str, list[int]],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
) -> None:
    group_names = list(index_groups.keys())
    ds = multiclass_group_discriminant(y_pred, [index_groups[g] for g in group_names])

    figs = []
    for i, group_name in enumerate(group_names):
        group_indices = index_groups[group_name]

        class_scores = ds[i]

        signal_mask = np.isin(y_true, group_indices)
        signal_scores = class_scores[signal_mask]

        background_mask = ~signal_mask
        background_scores = class_scores[background_mask]

        fig, ax = plt.subplots(figsize=(7, 6.25))

        signal_scores = iqr_remove_outliers(signal_scores) if signal_scores.size else signal_scores
        background_scores = iqr_remove_outliers(background_scores) if background_scores.size else background_scores

        if signal_scores.size and background_scores.size:
            signal_min_max = (min(signal_scores), max(signal_scores))
            background_min_max = (min(background_scores), max(background_scores))
            bins = np.linspace(
                min(signal_min_max[0], background_min_max[0]),
                max(signal_min_max[1], background_min_max[1]),
                nbins,
            )
        else:
            bins = np.linspace(-10, 10, nbins)

        ax.hist(
            signal_scores,
            bins=bins,
            histtype="step",
            label=f"Signal: {group_name}",
            lw=2.0,
            color="C0",
        )
        ax.hist(
            background_scores,
            bins=bins,
            histtype="step",
            label="Background: rest",
            lw=2.0,
            color="C1",
        )

        ax.set_xlabel(f"Group Discriminant for {group_name}", fontsize=16)
        ax.set_ylabel("Events", fontsize=16)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=12)
        atlas_label(ax, loc=0, fontsize=14)

        figs.append(fig)
        plt.close(fig)

    os.makedirs(save_path, exist_ok=True)
    with PdfPages(f"{save_path}/custom_discriminant_{save_postfix}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


@handle_plot_exception
def plot_group_one_vs_rest_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    group_name: str,
    group_members: list[str],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
) -> None:
    # Build group mask and scores
    name_to_idx = class_labels
    member_idx = [name_to_idx[m] for m in group_members if m in name_to_idx]
    if len(member_idx) == 0:
        return

    p_group = np.sum(y_pred[:, member_idx], axis=1)

    is_signal = np.isin(y_true, member_idx)

    score_sig = p_group[is_signal]
    score_bkg = p_group[~is_signal]

    bins = list(np.linspace(0, 1, nbins))

    fig, ax = plt.subplots(figsize=(7, 6.25))
    ax.hist(
        score_sig,
        bins=bins,
        histtype="step",
        lw=2.0,
        color="C0",
        density=True,
        label=f"Signal: {group_name}",
    )
    ax.hist(
        score_bkg,
        bins=bins,
        histtype="step",
        lw=2.0,
        color="C1",
        density=True,
        label="Background: rest",
    )

    ax.set_xlabel(f"Model Score for {group_name} (one-vs-rest)", fontsize=16)
    ax.set_ylabel("Events (normalized)", fontsize=16)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=12)
    atlas_label(ax, loc=0, fontsize=14)
    os.makedirs(f"{save_path}/custom_groups", exist_ok=True)
    save_plot(
        fig, f"{save_path}/custom_groups/one_vs_rest_group_score_{group_name}_{save_postfix}.pdf", use_format="pdf"
    )


@handle_plot_exception
def plot_group_one_vs_rest_roc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    group_name: str,
    group_members: list[str],
    save_path: str,
    save_postfix: str,
) -> None:
    name_to_idx = class_labels
    member_idx = [name_to_idx[m] for m in group_members if m in name_to_idx]
    if len(member_idx) == 0:
        return

    p_group = np.sum(y_pred[:, member_idx], axis=1)

    is_signal = np.isin(y_true, member_idx)

    # 1 for signal, 0 for rest
    y_true_binary = is_signal.astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, p_group)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6.25))
    ax.plot(fpr, tpr, label=f"{group_name} (AUC = {roc_auc:.4e})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC Curve (OVR: {group_name})", loc="right")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    atlas_label(ax, loc=0, fontsize=14)
    save_plot(fig, f"{save_path}/custom_groups/multiclass_group_roc_curve_{group_name}_{save_postfix}.pdf")


@handle_plot_exception
def plot_group_one_vs_rest_discriminant(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    group_name: str,
    group_members: list[str],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
) -> None:
    name_to_idx = class_labels
    member_idx = [name_to_idx[m] for m in group_members if m in name_to_idx]
    if len(member_idx) == 0:
        return

    p_group = np.sum(y_pred[:, member_idx], axis=1)
    p_rest = 1.0 - p_group
    disc = np.log((p_group + 1e-12) / (p_rest + 1e-12))

    is_signal = np.isin(y_true, member_idx)

    signal_scores = disc[is_signal]
    background_scores = disc[~is_signal]

    signal_scores = iqr_remove_outliers(signal_scores)
    background_scores = iqr_remove_outliers(background_scores)
    bins = np.linspace(
        min(signal_scores.min(), background_scores.min()), max(signal_scores.max(), background_scores.max()), nbins
    )

    fig, ax = plt.subplots(figsize=(7, 6.25))
    ax.hist(signal_scores, bins=bins, histtype="step", lw=2.0, color="C0", label=f"Signal: {group_name}")
    ax.hist(background_scores, bins=bins, histtype="step", lw=2.0, color="C1", label="Background: rest")

    ax.set_xlabel(f"Discriminant for {group_name} (log p_group/p_rest)", fontsize=16)
    ax.set_ylabel("Events", fontsize=16)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=12)
    atlas_label(ax, loc=0, fontsize=14)
    os.makedirs(f"{save_path}/custom_groups", exist_ok=True)
    save_plot(
        fig,
        f"{save_path}/custom_groups/one_vs_rest_group_discriminant_{group_name}_{save_postfix}.pdf",
        use_format="pdf",
    )
