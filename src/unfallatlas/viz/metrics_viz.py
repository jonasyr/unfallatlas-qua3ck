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
    title_prefix: str = "",
):
    """Overlay interactive Plotly ROC and PR curves for multiple (y_true, y_score) pairs.

    Args:
        models: maps a display name to a (y_true, y_score) tuple, where
            y_score is the predicted probability / decision score for the
            positive (KSI) class.
        title_prefix: Prepended to both plot titles.

    Returns:
        (roc_fig, pr_fig) - two `plotly.graph_objects.Figure` instances.
    """
    import plotly.graph_objects as go
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    roc_fig = go.Figure()
    pr_fig = go.Figure()

    for name, (y_true, y_score) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        roc_fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.3f})", hoverinfo="all"
            )
        )

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        pr_fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{name} (AUC={pr_auc:.3f})",
                hoverinfo="all",
            )
        )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray")
        )
    )
    roc_fig.update_layout(
        title=f"{title_prefix} ROC Curve".strip(),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend=dict(
            x=0.99, y=0.02, xanchor="right", yanchor="bottom", bgcolor="rgba(255,255,255,0.8)"
        ),
    )

    pr_fig.update_layout(
        title=f"{title_prefix} Precision-Recall Curve".strip(),
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        legend=dict(
            x=0.99, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.8)"
        ),
    )

    return roc_fig, pr_fig


def plot_confusion_matrix_heatmap(cm, labels: list[str], title: str = ""):
    """Annotated interactive Plotly confusion-matrix heatmap for a binary classifier."""
    import numpy as np
    import plotly.graph_objects as go

    cm = np.asarray(cm)
    x_labels = [f"Pred {label}" for label in labels]
    y_labels = [f"True {label}" for label in labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text:,}",
            textfont={"size": 14},
            hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(title=title, template="plotly_white", yaxis=dict(autorange="reversed"))
    return fig
