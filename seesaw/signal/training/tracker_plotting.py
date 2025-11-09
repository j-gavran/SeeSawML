import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from f9columnar.utils.helpers import handle_plot_exception
from matplotlib.backends.backend_pdf import PdfPages
from plothist import get_color_palette
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve

from seesaw.signal.utils import multiclass_discriminant
from seesaw.utils.labels import get_label
from seesaw.utils.plots_utils import atlas_label, get_color, iqr_remove_outliers, iqr_remove_outliers_mask, save_plot


@handle_plot_exception
def plot_binary_roc(fpr: np.ndarray, tpr: np.ndarray, au_roc: float, save_path: str, save_postfix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(fpr, tpr, color="C0", label="ROC curve", lw=2.5)
    ax.plot([0, 1], [0, 1], color="C1", linestyle="--", label="Random guess", lw=2.5)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title(f"AUROC={au_roc:.6e}", fontsize=16, loc="right")
    ax.legend(loc="lower right")

    atlas_label(ax, loc=0, fontsize=16)
    save_plot(fig, f"{save_path}/roc_curve_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_binary_bkg_rej_vs_sig_eff(fpr: np.ndarray, tpr: np.ndarray, save_path: str, save_postfix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(tpr, 1 / (fpr + 1e-6), color="C0", lw=2.5)
    ax.set_xlim((0.0, 1.0))
    ax.set_xlabel(r"Signal Efficiency $\varepsilon_\mathrm{sig}$", fontsize=16)
    ax.set_ylabel(r"Background Rejection $\varepsilon_\mathrm{bkg}^{-1}$", fontsize=16)

    ax.set_yscale("log")

    atlas_label(ax, loc=0, fontsize=16)
    save_plot(fig, f"{save_path}/bkg_rej_vs_sig_eff_curve_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_binary_precision_recall(
    precision: np.ndarray, recall: np.ndarray, au_prc: float, save_path: str, save_postfix: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(recall, precision, color="C0", lw=2.5)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_title(f"AUPRC={au_prc:.6e}", fontsize=16, loc="right")

    atlas_label(ax, loc=0, fontsize=16)
    save_plot(fig, f"{save_path}/precision_recall_curve_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_binary_confusion_matrix(
    bcm: np.ndarray, sig_label: int, bkg_label: int, save_path: str, save_postfix: str
) -> None:
    for to_perc in [True, False]:
        fig, ax = plt.subplots(figsize=(8, 7))

        if sig_label == 0:
            xticklabels = [f"Pred {sig_label}", f"Pred {bkg_label}"]
            yticklabels = [f"True {sig_label}", f"True {bkg_label}"]
        else:
            xticklabels = [f"Pred {bkg_label}", f"Pred {sig_label}"]
            yticklabels = [f"True {bkg_label}", f"True {sig_label}"]

        # 0, 0: True negatives
        # 0, 1: False positives
        # 1, 0: False negatives
        # 1, 1: True positives

        sns.heatmap(
            bcm / bcm.sum(axis=1, keepdims=True) if to_perc else bcm,
            annot=True,
            fmt=".2%" if to_perc else "d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="lightgray",
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            annot_kws={"fontsize": 14, "weight": "bold"},
        )

        ax.set_xlabel("Predicted Label", fontsize=16)
        ax.set_ylabel("True Label", fontsize=16)
        ax.set_title(f"Signal={sig_label}, Background={bkg_label}", fontsize=16, loc="right")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

        atlas_label(ax, loc=0, fontsize=16)
        if to_perc:
            save_plot(fig, f"{save_path}/confusion_matrix_{save_postfix}_perc.pdf", use_format="pdf")
        else:
            save_plot(fig, f"{save_path}/confusion_matrix_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_binary_model_score(
    score: np.ndarray,
    labels: np.ndarray,
    sig_label: int,
    bkg_label: int,
    n_bins: int,
    save_path: str,
    save_postfix: str,
    density: bool = False,
    mc_weights: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    sig_weights = None if mc_weights is None else mc_weights[labels == sig_label]
    bkg_weights = None if mc_weights is None else mc_weights[labels == bkg_label]

    score_sig = score[labels == sig_label]
    score_bkg = score[labels == bkg_label]

    if "sigmoid" not in save_postfix:
        sig_mask = iqr_remove_outliers_mask(score_sig)
        bkg_mask = iqr_remove_outliers_mask(score_bkg)

        score_sig = score_sig[sig_mask]
        score_bkg = score_bkg[bkg_mask]

        if sig_weights is not None and bkg_weights is not None:
            sig_weights = sig_weights[sig_mask]
            bkg_weights = bkg_weights[bkg_mask]

        x_min = min(np.min(score_sig), np.min(score_bkg))
        x_max = max(np.max(score_sig), np.max(score_bkg))
    else:
        x_min, x_max = 0.0, 1.0

    bins = list(np.linspace(x_min, x_max, n_bins))

    ax.hist(
        score_sig,
        bins=bins,
        histtype="step",
        color=get_color("Blue").rgb,
        label="Signal",
        lw=2,
        density=density,
        weights=sig_weights,
    )
    ax.hist(
        score_bkg,
        bins=bins,
        histtype="step",
        color=get_color("Red").rgb,
        label="Background",
        lw=2,
        density=density,
        weights=bkg_weights,
    )

    # ax.set_ylim(bottom=1.0)
    ax.set_yscale("log")

    ax.set_xlabel("Model Score", fontsize=16)

    ylabel = ""
    if mc_weights is not None:
        ylabel += "MC Weighted "

    if density:
        ylabel += "Normalised "

    ylabel += "Events"

    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(loc="upper right")

    atlas_label(ax, loc=0, fontsize=16)
    save_plot(fig, f"{save_path}/model_score_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_multiclass_confusion_matrix(cm: np.ndarray, labels: dict[str, int], save_path: str, save_postfix: str) -> None:
    labels_lst = list(labels.keys())

    for i, label in enumerate(labels_lst):
        labels_lst[i] = get_label(label).latex_name

    for to_perc in [True, False]:
        fig, ax = plt.subplots(figsize=(len(labels_lst) * 0.9, len(labels_lst) * 0.9))

        if to_perc:
            cm_perc = cm / cm.sum(axis=1, keepdims=True)
            cm_perc = np.nan_to_num(cm_perc, nan=0.0, posinf=0.0, neginf=0.0)

        sns.heatmap(
            cm_perc if to_perc else cm,
            annot=True,
            fmt=".2%" if to_perc else "d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="lightgray",
            square=True,
            xticklabels=labels_lst,
            yticklabels=labels_lst,
            annot_kws={"fontsize": 10, "weight": "bold"},
        )

        ax.set_xlabel("Predicted Label", fontsize=16)
        ax.set_ylabel("True Label", fontsize=16)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

        plt.xlabel("Predicted")
        plt.ylabel("True")

        atlas_label(ax, loc=0, fontsize=16)
        if to_perc:
            save_plot(fig, f"{save_path}/confusion_matrix_{save_postfix}_perc.pdf", use_format="pdf")
        else:
            save_plot(fig, f"{save_path}/confusion_matrix_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_multiclass_model_score(
    scores: np.ndarray,
    labels: dict[str, int],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
    is_softmax: bool = True,
) -> None:
    labels_lst = list(labels.keys())

    if is_softmax:
        bins = list(np.linspace(0, 1, nbins))

    fig, ax = plt.subplots(figsize=(8.25, 6.25))

    colors = get_color_palette("cubehelix", len(labels_lst) - 1)

    for i, label in enumerate(labels_lst):
        label = get_label(label).latex_name

        if not is_softmax:
            score = iqr_remove_outliers(scores[:, i])
            bins = list(np.linspace(np.min(score), np.max(score), nbins))
        else:
            score = scores[:, i]

        ax.hist(
            score,
            bins=bins,
            label=label,
            color="r" if i == 0 else colors[i - 1],
            lw=1.25,
            histtype="step",
            zorder=len(labels_lst) - i,
        )

    ax.set_xlabel("Model Score", fontsize=16)
    ax.set_ylabel("Events", fontsize=16)
    ax.set_yscale("log")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=12)

    atlas_label(ax, loc=0, fontsize=14)
    if is_softmax:
        save_plot(fig, f"{save_path}/multiclass_score_{save_postfix}.pdf", use_format="pdf")
    else:
        save_plot(fig, f"{save_path}/multiclass_logits_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_multiclass_tsne_pca(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    save_path: str,
    save_postfix: str,
    n_max: int = 4096,
    use_pca: bool = False,
) -> None:
    n_samples = min(n_max, len(X))

    X = X[:n_samples]
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]

    inverse_class_labels = {v: k for k, v in class_labels.items()}

    label_combos = []
    for t, p in zip(y_true, y_pred):
        true_label = get_label(inverse_class_labels[t]).latex_name
        pred_label = get_label(inverse_class_labels[p]).latex_name

        label_combos.append(f"{true_label} / {pred_label}")

    if use_pca:
        pca = PCA(n_components=2)
        X_embedded = pca.fit_transform(X)
    else:
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
        X_embedded = tsne.fit_transform(X)

    unique_combos = sorted(set(label_combos))
    palette = get_color_palette("cubehelix", len(unique_combos))
    combo_to_color = {combo: palette[i] for i, combo in enumerate(unique_combos)}

    markers = {}
    for combo in unique_combos:
        t, p = combo.split(" / ")
        if t == p:
            markers[combo] = "o"
        else:
            markers[combo] = "X"

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=label_combos,
        palette=combo_to_color,
        style=label_combos,
        markers=markers,
        ax=ax,
        s=60,
        linewidth=0.5,
        edgecolor="gray",
        alpha=0.9,
        legend="brief",
    )

    if use_pca:
        ax.set_title("PCA: True vs. Predicted Classes", fontsize=18, loc="right")
    else:
        ax.set_title("t-SNE: True vs. Predicted Classes", fontsize=18, loc="right")

    ax.set_xlabel("Component 1", fontsize=18)
    ax.set_ylabel("Component 2", fontsize=18)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, title="True / Predicted", fontsize=16)

    atlas_label(ax, loc=0, fontsize=16)

    if use_pca:
        save_plot(fig, f"{save_path}/pca_{save_postfix}.pdf", use_format="pdf")
    else:
        save_plot(fig, f"{save_path}/tsne_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_multiclass_one_vs_rest_roc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    save_path: str,
    save_postfix: str,
) -> None:
    num_classes = len(class_labels)
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    inverse_class_labels = {v: k for k, v in class_labels.items()}

    colors = get_color_palette("cubehelix", num_classes - 1)

    fig, ax = plt.subplots(figsize=(7, 6.25))

    for i in range(num_classes):
        label = get_label(inverse_class_labels[i]).latex_name

        if i == 0:
            c, ls = "r", "--"
        else:
            c, ls = colors[i - 1], "-"

        ax.plot(fpr[i], tpr[i], label=f"{label} (AUC = {roc_auc[i]:.4e})", c=c, ls=ls, zorder=num_classes - i)

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve (One-vs-Rest)", loc="right")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    atlas_label(ax, loc=0, fontsize=14)

    save_plot(fig, f"{save_path}/multiclass_roc_curve_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_multiclass_discriminant(
    y_pred: np.ndarray, class_labels: dict[str, int], save_path: str, save_postfix: str, nbins: int = 60
) -> None:
    ds = multiclass_discriminant(y_pred)

    labels = list(class_labels.keys())
    colors = get_color_palette("cubehelix", len(labels) - 1)

    fig, ax = plt.subplots(figsize=(8.25, 6.25))

    for i, label in enumerate(labels):
        label = get_label(label).latex_name

        if i == 0:
            c = "r"
        else:
            c = colors[i - 1]

        ds[i] = iqr_remove_outliers(ds[i])
        bins = np.linspace(min(ds[i]), max(ds[i]), nbins)

        ax.hist(
            ds[i],
            bins=bins,
            histtype="step",
            label=label,
            lw=1.25,
            alpha=0.7,
            color=c,
            zorder=len(labels) - i,
        )

    ax.set_xlabel("Discriminant Score", fontsize=16)
    ax.set_ylabel("Events", fontsize=16)
    ax.set_yscale("log")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=12)
    atlas_label(ax, loc=0, fontsize=14)
    save_plot(fig, f"{save_path}/multiclass_log_discriminant_{save_postfix}.pdf", use_format="pdf")


@handle_plot_exception
def plot_multiclass_one_vs_rest_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
) -> None:
    inverse_class_labels = {v: k for k, v in class_labels.items()}
    plot_labels = [get_label(inverse_class_labels[i]).latex_name for i in range(len(class_labels))]

    bins = list(np.linspace(0, 1, nbins))

    y_true_class_indices = np.argmax(y_true, axis=1)

    figs = []
    for class_idx in range(len(class_labels)):
        signal_label = plot_labels[class_idx]
        background_labels = set(plot_labels) - {signal_label}

        class_scores = y_pred[:, class_idx]

        signal_mask = y_true_class_indices == class_idx
        signal_scores = class_scores[signal_mask]

        background_mask = y_true_class_indices != class_idx
        background_scores = class_scores[background_mask]

        fig, ax = plt.subplots(figsize=(7, 6.25))

        ax.hist(
            signal_scores,
            bins=bins,
            histtype="step",
            label=f"Signal: {signal_label} (One-vs-Rest)",
            lw=2.0,
            color=get_color("Blue").rgb,
            density=True,
        )
        ax.hist(
            background_scores,
            bins=bins,
            histtype="step",
            label=f"Background: {', '.join(background_labels)}",
            lw=2.0,
            color=get_color("Red").rgb,
            density=True,
        )

        signal_min_max = (min(signal_scores), max(signal_scores))
        ax.text(
            0.05,
            0.05,
            f"Signal: {signal_label}\nMin: {signal_min_max[0]:.3f}, Max: {signal_min_max[1]:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
        )

        ax.set_xlabel(f"Model Score for {signal_label}", fontsize=16)
        ax.set_ylabel("Events (normalized)", fontsize=16)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=12)
        atlas_label(ax, loc=0, fontsize=14)

        figs.append(fig)
        plt.close(fig)

    with PdfPages(f"{save_path}/one_vs_rest_score_{save_postfix}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")


@handle_plot_exception
def plot_multiclass_discriminant_one_vs_rest(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    class_labels: dict[str, int],
    save_path: str,
    save_postfix: str,
    nbins: int = 100,
) -> None:
    ds = multiclass_discriminant(y_pred)

    inverse_class_labels = {v: k for k, v in class_labels.items()}
    labels = [get_label(inverse_class_labels[i]).latex_name for i in range(len(class_labels))]

    y_true_class_indices = np.argmax(y_true, axis=1)

    figs = []
    for i, label in enumerate(labels):
        signal_label = label
        background_labels = set(labels) - {signal_label}

        class_scores = ds[i]

        signal_mask = y_true_class_indices == i
        signal_scores = class_scores[signal_mask]

        background_mask = y_true_class_indices != i
        background_scores = class_scores[background_mask]

        fig, ax = plt.subplots(figsize=(7, 6.25))

        signal_scores = iqr_remove_outliers(signal_scores)
        background_scores = iqr_remove_outliers(background_scores)

        signal_min_max = (min(signal_scores), max(signal_scores))
        background_min_max = (min(background_scores), max(background_scores))

        bins = np.linspace(
            min(signal_min_max[0], background_min_max[0]),
            max(signal_min_max[1], background_min_max[1]),
            nbins,
        )

        ax.hist(
            signal_scores,
            bins=bins,
            histtype="step",
            label=f"Signal: {signal_label} (One-vs-Rest)",
            lw=2.0,
            color=get_color("Blue").rgb,
        )
        ax.hist(
            background_scores,
            bins=bins,
            histtype="step",
            label=f"Background: {', '.join(background_labels)}",
            lw=2.0,
            color=get_color("Red").rgb,
        )

        ax.text(
            0.05,
            0.05,
            f"Signal: {signal_label}\nMin: {signal_min_max[0]:.3f}, Max: {signal_min_max[1]:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
        )

        ax.set_xlabel(f"Discriminant Score for {signal_label}", fontsize=16)
        ax.set_ylabel("Events", fontsize=16)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=12)
        atlas_label(ax, loc=0, fontsize=14)

        figs.append(fig)
        plt.close(fig)

    with PdfPages(f"{save_path}/one_vs_rest_discriminant_{save_postfix}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
