"""Diagnostic plots for model selection and Pareto-front analysis."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _plot_pareto_front(
    comparison_df: pd.DataFrame,
    *,
    recall_col: str,
    recall_axis_label: str,
    title: str,
    gate_f1: float,
    gate_recall: float,
    label_col: str,
) -> go.Figure:
    """Create an interactive macro-F1 versus recall Pareto-front chart."""
    fig = go.Figure(
        go.Scatter(
            x=comparison_df[recall_col],
            y=comparison_df["macro_f1"],
            mode="markers+text",
            text=comparison_df[label_col],
            textposition="top center",
            customdata=comparison_df[[label_col]].to_numpy(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + recall_axis_label
                + ": %{x:.3f}<br>Macro-F1: %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_shape(
        type="rect",
        x0=gate_recall,
        x1=1,
        y0=gate_f1,
        y1=1,
        fillcolor="rgba(42, 157, 143, 0.12)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=gate_recall, line_dash="dash", line_color="#E63946")
    fig.add_hline(y=gate_f1, line_dash="dash", line_color="#E63946")
    fig.update_layout(
        title=title,
        xaxis_title=recall_axis_label,
        yaxis_title="Macro-F1",
        template="plotly_white",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig


def plot_f1_recall_front(
    comparison_df: pd.DataFrame,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
) -> go.Figure:
    """Scatter macro-F1 vs. Recall(class 1) for every 3-class model configuration.

    Draws dashed gate lines at gate_f1 and gate_recall; shades the feasible
    quadrant (top-right). Use this to show the gate is outside the empirical
    Pareto front for the 3-class problem.

    Args:
        comparison_df: DataFrame with columns [label_col, 'macro_f1', 'recall_class_1'].
        gate_f1: Horizontal gate line for macro-F1.
        gate_recall: Vertical gate line for Recall(class 1).
        label_col: Column used to label each point.

    Returns:
        Interactive Plotly figure.
    """
    return _plot_pareto_front(
        comparison_df,
        recall_col="recall_class_1",
        recall_axis_label="Recall (Class 1 — Killed)",
        title="Pareto Front: Macro-F1 vs. Recall(Killed) — all 19 configurations",
        gate_f1=gate_f1,
        gate_recall=gate_recall,
        label_col=label_col,
    )


def plot_binary_f1_recall_front(
    comparison_df: pd.DataFrame,
    gate_f1: float = 0.55,
    gate_recall: float = 0.50,
    label_col: str = "model",
    title: str = "Pareto Front: Macro-F1 vs. Recall(KSI) — binary KSI candidates",
) -> go.Figure:
    """Scatter macro-F1 vs. Recall(KSI) for every binary-KSI model configuration.

    Analogous to plot_f1_recall_front, but reads the binary-evaluation
    column name ('recall_ksi', from evaluate_binary_predictions) instead of
    the 3-class 'recall_class_1'. Use this to compare the LightGBM binary
    champion against SVM (and any future) candidate families on the same
    axes as the revised binary gate.

    Args:
        comparison_df: DataFrame with columns [label_col, 'macro_f1', 'recall_ksi'].
        gate_f1: Horizontal gate line for macro-F1.
        gate_recall: Vertical gate line for Recall(KSI).
        label_col: Column used to label each point.
        title: Plot title (override when the candidate count/composition changes).

    Returns:
        Interactive Plotly figure.
    """
    return _plot_pareto_front(
        comparison_df,
        recall_col="recall_ksi",
        recall_axis_label="Recall (KSI)",
        title=title,
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
