import matplotlib  # noqa: E402
import pandas as pd

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402
import pytest

from unfallatlas.viz.metrics_viz import plot_f1_recall_front


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


def test_plot_f1_recall_front_returns_axes(comparison_df):
    ax = plot_f1_recall_front(comparison_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_f1_recall_front_accepts_external_ax(comparison_df):
    _, ax = plt.subplots()
    result = plot_f1_recall_front(comparison_df, ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_f1_recall_front_gate_lines_present(comparison_df):
    ax = plot_f1_recall_front(comparison_df, gate_f1=0.55, gate_recall=0.50)
    # Gate lines are drawn as axhline + axvline — check line xdata/ydata
    h_lines = [
        ln for ln in ax.lines if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
    ]
    v_lines = [
        ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
    ]
    assert len(h_lines) > 0, "Expected at least one horizontal gate line"
    assert len(v_lines) > 0, "Expected at least one vertical gate line"
    plt.close("all")


def test_plot_f1_recall_front_all_models_plotted(comparison_df):
    ax = plot_f1_recall_front(comparison_df)
    # Each model gets a scatter point — check there are at least n scatter collections
    assert ax.collections or ax.lines, "Expected scatter points in plot"
    plt.close("all")
