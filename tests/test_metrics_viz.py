import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from unfallatlas.viz.metrics_viz import (
    plot_binary_f1_recall_front,
    plot_confusion_matrix_heatmap,
    plot_f1_recall_front,
    plot_roc_pr_curves,
)


@pytest.fixture()
def comparison_df():
    return pd.DataFrame(
        [
            {"model": "lgbm_balanced", "macro_f1": 0.372, "recall_class_1": 0.621},
            {"model": "rf_balanced", "macro_f1": 0.424, "recall_class_1": 0.212},
            {"model": "logistic_reg", "macro_f1": 0.352, "recall_class_1": 0.641},
            {"model": "catboost_balanced", "macro_f1": 0.370, "recall_class_1": 0.571},
        ]
    )


def test_plot_f1_recall_front_returns_interactive_plotly_figure(comparison_df):
    fig = plot_f1_recall_front(comparison_df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert len(fig.data[0].x) == len(comparison_df)
    assert {shape.type for shape in fig.layout.shapes} == {"line", "rect"}


@pytest.fixture()
def binary_comparison_df():
    return pd.DataFrame(
        [
            {
                "model": "lightgbm_binary_balanced (champion)",
                "macro_f1": 0.6069,
                "recall_ksi": 0.5233,
            },
            {"model": "svm_linear_C1.0", "macro_f1": 0.55, "recall_ksi": 0.40},
            {"model": "svm_rbf_C1.0_gammascale", "macro_f1": 0.50, "recall_ksi": 0.35},
            {"model": "svm_sgd_hinge_alpha0.0001", "macro_f1": 0.52, "recall_ksi": 0.45},
        ]
    )


def test_plot_binary_front_uses_recall_ksi(binary_comparison_df):
    fig = plot_binary_f1_recall_front(binary_comparison_df)

    assert list(fig.data[0].x) == list(binary_comparison_df["recall_ksi"])
    assert list(fig.data[0].y) == list(binary_comparison_df["macro_f1"])


def test_plot_roc_pr_curves_returns_two_plotly_figures():
    import plotly.graph_objects as go

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    models = {
        "champion": (y_true, rng.random(200)),
        "runner_up": (y_true, rng.random(200)),
    }
    roc_fig, pr_fig = plot_roc_pr_curves(models, title_prefix="Test")
    assert isinstance(roc_fig, go.Figure)
    assert isinstance(pr_fig, go.Figure)
    assert len(roc_fig.data) == 3  # 2 model curves + chance line
    assert len(pr_fig.data) == 2
    assert roc_fig.layout.title.text == "Test ROC Curve"
    assert pr_fig.layout.title.text == "Test Precision-Recall Curve"


def test_plot_confusion_matrix_heatmap_returns_plotly_figure():
    import plotly.graph_objects as go

    cm = np.array([[23228, 20970], [53506, 170815]])
    fig = plot_confusion_matrix_heatmap(cm, labels=["KSI", "slight"], title="Test CM")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test CM"
    assert fig.data[0].z.tolist() == cm.tolist()
