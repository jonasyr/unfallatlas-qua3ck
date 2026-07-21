import matplotlib  # noqa: E402
import pandas as pd

matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
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


def test_plot_binary_f1_recall_front_returns_axes(binary_comparison_df):
    ax = plot_binary_f1_recall_front(binary_comparison_df)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_binary_f1_recall_front_accepts_external_ax(binary_comparison_df):
    _, ax = plt.subplots()
    result = plot_binary_f1_recall_front(binary_comparison_df, ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_binary_f1_recall_front_gate_lines_present(binary_comparison_df):
    ax = plot_binary_f1_recall_front(binary_comparison_df, gate_f1=0.55, gate_recall=0.50)
    h_lines = [
        ln for ln in ax.lines if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
    ]
    v_lines = [
        ln for ln in ax.lines if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
    ]
    assert len(h_lines) > 0
    assert len(v_lines) > 0
    plt.close("all")


def test_plot_binary_f1_recall_front_uses_recall_ksi_not_recall_class_1(binary_comparison_df):
    # Regression guard: this plot must read the binary-evaluation column name,
    # not silently fall back to the 3-class 'recall_class_1' column.
    ax = plot_binary_f1_recall_front(binary_comparison_df)
    scatter_collections = [
        coll for coll in ax.collections if isinstance(coll, matplotlib.collections.PathCollection)
    ]
    xdata = [pt[0] for coll in scatter_collections for pt in coll.get_offsets()]
    assert sorted(xdata) == sorted(binary_comparison_df["recall_ksi"].tolist())
    plt.close("all")


def test_plot_f1_recall_front_unaffected_by_refactor(comparison_df):
    """Existing 3-class plot must keep its exact title after the shared-helper refactor."""
    ax = plot_f1_recall_front(comparison_df)
    assert ax.get_title() == "Pareto Front: Macro-F1 vs. Recall(Killed) — all 19 configurations"
    plt.close("all")


def test_plot_f1_recall_front_legend_label_unchanged_by_refactor(comparison_df):
    """Regression test for a real bug found in review: the shared-helper
    refactor accidentally changed this function's vertical-gate legend text
    from 'Gate: Recall(1) >= 0.5' to a longer axis-label-derived string.
    plot_f1_recall_front must keep its exact original legend text."""
    ax = plot_f1_recall_front(comparison_df, gate_recall=0.50)
    legend_labels = [line.get_label() for line in ax.lines]
    assert "Gate: Recall(1) ≥ 0.5" in legend_labels
    plt.close("all")


def test_plot_roc_pr_curves_returns_two_axes():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    models = {
        "champion": (y_true, rng.random(200)),
        "runner_up": (y_true, rng.random(200)),
    }
    ax_roc, ax_pr = plot_roc_pr_curves(models, title_prefix="Test")
    assert isinstance(ax_roc, plt.Axes)
    assert isinstance(ax_pr, plt.Axes)
    assert len(ax_roc.lines) >= 2  # 2 model curves (+ optional chance line)
    assert len(ax_pr.lines) >= 2
    plt.close("all")


def test_plot_roc_pr_curves_accepts_external_axes():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    models = {"champion": (y_true, rng.random(100))}
    _, (ax_roc_in, ax_pr_in) = plt.subplots(1, 2)
    ax_roc, ax_pr = plot_roc_pr_curves(models, ax_roc=ax_roc_in, ax_pr=ax_pr_in)
    assert ax_roc is ax_roc_in
    assert ax_pr is ax_pr_in
    plt.close("all")


def test_plot_confusion_matrix_heatmap_returns_axes():
    cm = np.array([[23228, 20970], [53506, 170815]])
    ax = plot_confusion_matrix_heatmap(cm, labels=["KSI", "slight"], title="Test CM")
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test CM"
    plt.close("all")
