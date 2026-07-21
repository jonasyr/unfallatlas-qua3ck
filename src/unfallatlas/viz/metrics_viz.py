"""Diagnostic plots for model selection and Pareto-front analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def _plot_pareto_front(
    comparison_df: pd.DataFrame,
    recall_col: str,
    recall_axis_label: str,
    recall_gate_label: str,
    title: str,
    ax: plt.Axes | None,
    gate_f1: float,
    gate_recall: float,
    label_col: str,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        comparison_df[recall_col],
        comparison_df["macro_f1"],
        zorder=3,
        s=60,
        color="steelblue",
    )

    for _, row in comparison_df.iterrows():
        ax.annotate(
            row[label_col],
            xy=(row[recall_col], row["macro_f1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(
        gate_f1, color="crimson", linestyle="--", linewidth=1.2, label=f"Gate: macro-F1 ≥ {gate_f1}"
    )
    ax.axvline(
        gate_recall,
        color="darkorange",
        linestyle="--",
        linewidth=1.2,
        label=f"Gate: {recall_gate_label} ≥ {gate_recall}",
    )

    ax.fill_between(
        [gate_recall, ax.get_xlim()[1] if ax.get_xlim()[1] > gate_recall else 1.0],
        gate_f1,
        1.0,
        alpha=0.08,
        color="green",
        label="Feasible zone",
    )

    ax.set_xlabel(recall_axis_label, fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    return ax


def plot_f1_recall_front(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
) -> plt.Axes:
    """Scatter macro-F1 vs. Recall(class 1) for every 3-class model configuration.

    Draws dashed gate lines at gate_f1 and gate_recall; shades the feasible
    quadrant (top-right). Use this to show the gate is outside the empirical
    Pareto front for the 3-class problem.

    Args:
        comparison_df: DataFrame with columns [label_col, 'macro_f1', 'recall_class_1'].
        ax: Optional existing Axes; a new figure+axes is created if None.
        gate_f1: Horizontal gate line for macro-F1.
        gate_recall: Vertical gate line for Recall(class 1).
        label_col: Column used to label each point.

    Returns:
        The populated Axes object.
    """
    return _plot_pareto_front(
        comparison_df,
        recall_col="recall_class_1",
        recall_axis_label="Recall (Class 1 — Killed)",
        recall_gate_label="Recall(1)",
        title="Pareto Front: Macro-F1 vs. Recall(Killed) — all 19 configurations",
        ax=ax,
        gate_f1=gate_f1,
        gate_recall=gate_recall,
        label_col=label_col,
    )


def plot_binary_f1_recall_front(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
    title: str = "Pareto Front: Macro-F1 vs. Recall(KSI) — binary KSI candidates",
) -> plt.Axes:
    """Scatter macro-F1 vs. Recall(KSI) for every binary-KSI model configuration.

    Analogous to plot_f1_recall_front, but reads the binary-evaluation
    column name ('recall_ksi', from evaluate_binary_predictions) instead of
    the 3-class 'recall_class_1'. Use this to compare the LightGBM binary
    champion against SVM (and any future) candidate families on the same
    axes as the revised binary gate.

    Args:
        comparison_df: DataFrame with columns [label_col, 'macro_f1', 'recall_ksi'].
        ax: Optional existing Axes; a new figure+axes is created if None.
        gate_f1: Horizontal gate line for macro-F1.
        gate_recall: Vertical gate line for Recall(KSI).
        label_col: Column used to label each point.
        title: Plot title (override when the candidate count/composition changes).

    Returns:
        The populated Axes object.
    """
    return _plot_pareto_front(
        comparison_df,
        recall_col="recall_ksi",
        recall_axis_label="Recall (KSI)",
        recall_gate_label="Recall(KSI)",
        title=title,
        ax=ax,
        gate_f1=gate_f1,
        gate_recall=gate_recall,
        label_col=label_col,
    )


def plot_roc_pr_curves(
    models: dict[str, tuple],
    ax_roc: plt.Axes | None = None,
    ax_pr: plt.Axes | None = None,
    title_prefix: str = "",
) -> tuple[plt.Axes, plt.Axes]:
    """Overlay ROC and PR curves for multiple (y_true, y_score) pairs.

    Args:
        models: maps a display name to a (y_true, y_score) tuple, where
            y_score is the predicted probability / decision score for the
            positive (KSI) class.
        ax_roc: Optional existing Axes for the ROC curve; created if None.
        ax_pr: Optional existing Axes for the PR curve; created if None.
        title_prefix: Prepended to both plot titles.

    Returns:
        (ax_roc, ax_pr) — the populated Axes objects.
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    if ax_roc is None:
        _, ax_roc = plt.subplots()
    if ax_pr is None:
        _, ax_pr = plt.subplots()

    for name, (y_true, y_score) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{title_prefix} ROC Curve".strip())
    ax_roc.legend()

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"{title_prefix} Precision-Recall Curve".strip())
    ax_pr.legend()

    return ax_roc, ax_pr


def plot_confusion_matrix_heatmap(
    cm,
    labels: list[str],
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    """Annotated confusion-matrix heatmap for a binary classifier."""
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()

    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"Pred {label}" for label in labels])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"True {label}" for label in labels])

    cm_max = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm_max / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)

    ax.set_title(title)
    return ax
