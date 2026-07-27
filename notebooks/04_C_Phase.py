# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: unfallatlas-qua3ck
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Unfallatlas Germany: C Phase
#
# ## 0 Scope, artifact registry, and evaluation contract
#
# The C phase concludes the binary KSI analysis. KSI combines fatal and serious injury accidents, while the comparison class contains slight injury accidents.
#
# This phase does not train or tune another model. It validates the persisted A³ candidate registry, measures every candidate on Val 2023, compares the four substantive tree ensemble finalists, and reserves Test 2024 for one confirmation of the previously selected Random Forest champion.
#
# The evaluation contract is fixed before any test result is inspected:
#
# 1. All ten persisted candidates are compared only on Val 2023.
# 2. Random Forest, XGBoost, LightGBM, and CatBoost receive the detailed finalist analysis.
# 3. Test 2024 is used only for the preselected Random Forest champion at the validation threshold.
# 4. Cross model interpretation uses permutation importance on Val 2023.
# 5. SHAP explains only the champion and does not stand in for cross model evidence.

# %%
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shap
from sklearn.metrics import confusion_matrix

from unfallatlas.features.preprocessing import (
    chronological_split,
    load_training_frame,
    split_features_target_binary,
)
from unfallatlas.models.artifacts import validate_candidate_registry
from unfallatlas.models.c_phase import (
    build_inference_contract,
    build_qualitative_matrix,
    compute_error_slices,
)
from unfallatlas.models.candidate_analysis import (
    analysis_fingerprint,
    compute_finalist_permutation_importance,
    load_or_analyze_candidates,
    prediction_disagreement,
)
from unfallatlas.models.evaluate import evaluate_binary_predictions
from unfallatlas.viz.metrics_viz import (
    plot_binary_f1_recall_front,
    plot_confusion_matrix_heatmap,
    plot_roc_pr_curves,
)

pio.renderers.default = "plotly_mimetype"
pd.set_option("display.max_columns", None)
np.random.seed(42)

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_LABELS = {
    "binary_random_guess": "Random guess",
    "binary_majority_class": "Majority class",
    "binary_logistic_regression": "Logistic regression",
    "binary_random_forest_balanced": "Random Forest",
    "binary_xgboost_balanced": "XGBoost",
    "binary_lightgbm_balanced": "LightGBM",
    "binary_catboost_balanced": "CatBoost",
    "binary_svm_linear_balanced": "Linear SVM",
    "binary_svm_sgd_balanced": "SGD hinge SVM",
    "binary_svm_rbf_balanced": "RBF SVM",
}
FINALIST_FAMILIES = ["random_forest", "xgboost", "lightgbm", "catboost"]
RANDOM_STATE = 42
LATENCY_SAMPLE_SIZE = 2_000
ROBUSTNESS_SAMPLE_SIZE = 2_000
PERMUTATION_SAMPLE_SIZE = 2_000
PERMUTATION_REPEATS = 3
LATENCY_REPEATS = 5
ROBUSTNESS_COLUMNS = [
    "osm_road_density",
    "osm_way_count",
    "osm_maxspeed_mean",
    "dwd_temp_air_2m",
    "dwd_precip_mm",
]

# %%
model_card_path = PROCESSED_DIR / "a3_binary_model_card.json"
with model_card_path.open(encoding="utf-8") as file:
    model_card = json.load(file)

candidate_registry = model_card["candidate_artifacts"]
candidate_artifacts = validate_candidate_registry(candidate_registry, BASE_DIR)
artifact_by_model = {artifact.model: artifact for artifact in candidate_artifacts}
champion_artifact = next(
    artifact for artifact in candidate_artifacts if artifact.evaluation_role == "champion"
)

assert len(candidate_artifacts) == 10
assert {
    artifact.family
    for artifact in candidate_artifacts
    if artifact.evaluation_role in {"champion", "finalist"}
} == set(FINALIST_FAMILIES)
assert champion_artifact.family == model_card["champion_family"]

registry_fig = go.Figure(
    data=[
        go.Table(
            header={
                "values": ["Candidate", "Family", "Role", "Training rows", "Score interface"],
                "fill_color": "#14213D",
                "font": {"color": "white"},
                "align": "left",
            },
            cells={
                "values": [
                    [MODEL_LABELS[artifact.model] for artifact in candidate_artifacts],
                    [artifact.family for artifact in candidate_artifacts],
                    [artifact.evaluation_role for artifact in candidate_artifacts],
                    [
                        artifact.n_train if artifact.n_train is not None else "Not applicable"
                        for artifact in candidate_artifacts
                    ],
                    [artifact.score_interface for artifact in candidate_artifacts],
                ],
                "fill_color": "#F7F8FA",
                "align": "left",
            },
        )
    ]
)
registry_fig.update_layout(
    title="Validated persisted candidate registry",
    height=430,
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
registry_fig.show()

# %%
full_frame = load_training_frame(BASE_DIR)
train_frame, validation_frame, test_frame = chronological_split(full_frame)
X_train, y_train = split_features_target_binary(train_frame)
X_validation, y_validation = split_features_target_binary(validation_frame)
X_test, y_test = split_features_target_binary(test_frame)

feature_columns = list(X_train.columns)
feature_dtypes = {column: str(dtype) for column, dtype in X_train.dtypes.items()}
inference_contract = build_inference_contract(
    feature_columns,
    feature_dtypes,
    model_card,
    X_train,
)
del full_frame, train_frame, validation_frame, X_train, y_train

validation_hasher = hashlib.sha256()
validation_hasher.update(pd.util.hash_pandas_object(X_validation, index=True).to_numpy().tobytes())
validation_hasher.update(pd.util.hash_pandas_object(y_validation, index=True).to_numpy().tobytes())
validation_fingerprint = validation_hasher.hexdigest()

sample_rng = np.random.default_rng(RANDOM_STATE)
sample_indices = np.arange(len(X_validation))
latency_indices = sample_rng.choice(
    sample_indices,
    size=min(LATENCY_SAMPLE_SIZE, len(sample_indices)),
    replace=False,
)
robustness_indices = sample_rng.choice(
    sample_indices,
    size=min(ROBUSTNESS_SAMPLE_SIZE, len(sample_indices)),
    replace=False,
)

analysis_parameters = {
    "random_state": RANDOM_STATE,
    "latency_repeats": LATENCY_REPEATS,
    "latency_sample_size": int(len(latency_indices)),
    "robustness_sample_size": int(len(robustness_indices)),
    "robustness_columns": ROBUSTNESS_COLUMNS,
    "permutation_sample_size": PERMUTATION_SAMPLE_SIZE,
    "permutation_repeats": PERMUTATION_REPEATS,
}
candidate_analysis_fingerprint = analysis_fingerprint(
    candidate_artifacts,
    validation_fingerprint,
    analysis_parameters,
)
candidate_analysis = load_or_analyze_candidates(
    candidate_artifacts,
    X_val=X_validation,
    y_val=y_validation,
    cache_dir=PROCESSED_DIR,
    data_fingerprint=validation_fingerprint,
    parameters=analysis_parameters,
    latency_sample=X_validation.iloc[latency_indices],
    robustness_sample=X_validation.iloc[robustness_indices],
)

candidate_metrics = candidate_analysis.metrics.copy()
candidate_metrics["display_name"] = candidate_metrics["model"].map(MODEL_LABELS)
assert len(candidate_metrics) == len(candidate_artifacts)
assert set(candidate_metrics["model"]) == set(artifact_by_model)

# %% [markdown]
# ## 1 All candidate Val 2023 comparison
#
# The first view compares every persisted candidate at its saved operating point. Macro F1 balances performance across KSI and slight injury accidents. Recall KSI shows how many actual KSI cases are detected. The shaded gate requires macro F1 of at least 0.55 and Recall KSI of at least 0.50.

# %%
candidate_table = candidate_metrics.sort_values(
    ["macro_f1", "recall_ksi"], ascending=False
).reset_index(drop=True)

comparison_table_fig = go.Figure(
    data=[
        go.Table(
            header={
                "values": [
                    "Candidate",
                    "Role",
                    "Macro F1",
                    "Recall KSI",
                    "Recall slight",
                    "Latency ms per 1,000 rows",
                ],
                "fill_color": "#14213D",
                "font": {"color": "white"},
                "align": "left",
            },
            cells={
                "values": [
                    candidate_table["display_name"],
                    candidate_table["evaluation_role"],
                    candidate_table["macro_f1"].map(lambda value: f"{value:.3f}"),
                    candidate_table["recall_ksi"].map(lambda value: f"{value:.3f}"),
                    candidate_table["recall_slight"].map(lambda value: f"{value:.3f}"),
                    candidate_table["latency_ms_per_1k"].map(lambda value: f"{value:.2f}"),
                ],
                "fill_color": "#F7F8FA",
                "align": "left",
            },
        )
    ]
)
comparison_table_fig.update_layout(
    title="All persisted candidates measured on Val 2023",
    height=430,
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
comparison_table_fig.show()

front_fig = plot_binary_f1_recall_front(
    candidate_metrics,
    label_col="display_name",
    title="Val 2023 macro F1 and KSI recall",
)
front_fig.update_layout(height=560)
front_fig.show()

# %% [markdown]
# The candidate field is not a single ranking. Random Forest emphasizes balanced class performance, while the boosting finalists move further toward KSI recall. The operating point therefore matters as much as the model family.

# %% [markdown]
# ## 2 ROC, precision recall, and operating point tradeoffs
#
# Threshold free curves test whether a candidate ranks KSI cases ahead of slight injury cases across many possible thresholds. Baselines are omitted from this view because they do not provide a substantive learned ranking. The front in Section 1 then shows the actual saved operating point used for each candidate.

# %%
substantive_artifacts = [
    artifact for artifact in candidate_artifacts if artifact.evaluation_role not in {"baseline"}
]
curve_inputs = {
    MODEL_LABELS[artifact.model]: (
        y_validation.to_numpy(),
        candidate_analysis.scores[artifact.model],
    )
    for artifact in substantive_artifacts
}
roc_fig, precision_recall_fig = plot_roc_pr_curves(
    curve_inputs,
    title_prefix="Val 2023",
)
roc_fig.update_layout(title="Val 2023 ROC curves", height=600)
precision_recall_fig.update_layout(
    title="Val 2023 precision and recall curves",
    height=600,
)
roc_fig.show()
precision_recall_fig.show()

# %% [markdown]
# These curves separate ranking quality from threshold choice. A model can rank cases well but still occupy a different macro F1 and recall position after a threshold is applied. The finalist comparison therefore keeps both views visible.

# %% [markdown]
# ## 3 Measured finalist comparison
#
# Random Forest, XGBoost, LightGBM, and CatBoost are measured independently. Latency is the warmed median in milliseconds per 1,000 rows. Robustness sets one weather or road context feature to missing at a time, then records prediction failure, mean absolute score drift, and changed class share. No value is copied from another model.

# %%
finalist_artifacts = [
    artifact for artifact in candidate_artifacts if artifact.family in FINALIST_FAMILIES
]
finalist_models = [artifact.model for artifact in finalist_artifacts]
finalist_metrics = candidate_metrics[candidate_metrics["model"].isin(finalist_models)].copy()

robustness_detail = candidate_analysis.robustness[
    candidate_analysis.robustness["model"].isin(finalist_models)
].copy()
robustness_summary = robustness_detail.groupby("model", as_index=False).agg(
    failed_probes=("prediction_failed", "sum"),
    mean_abs_score_drift=("mean_abs_score_drift", "mean"),
    max_abs_score_drift=("mean_abs_score_drift", "max"),
    mean_changed_class_share=("changed_class_share", "mean"),
    max_changed_class_share=("changed_class_share", "max"),
)
finalist_summary = finalist_metrics.merge(robustness_summary, on="model", how="left")
finalist_summary["failed_probes"] = finalist_summary["failed_probes"].astype(int)
finalist_summary["robustness_status"] = np.where(
    finalist_summary["failed_probes"].gt(0),
    "Prediction failure observed",
    "All probes passed",
)
finalist_summary["robustness_score"] = np.where(
    finalist_summary["failed_probes"].gt(0),
    0.0,
    1.0 - finalist_summary["mean_changed_class_share"],
)
finalist_summary = finalist_summary.sort_values("macro_f1", ascending=False)

robustness_status_fig = go.Figure(
    data=[
        go.Table(
            header={
                "values": [
                    "Model",
                    "Probe status",
                    "Failed probes",
                    "Mean score drift",
                    "Maximum score drift",
                    "Mean changed class share",
                    "Maximum changed class share",
                ],
                "fill_color": "#14213D",
                "font": {"color": "white"},
                "align": "left",
            },
            cells={
                "values": [
                    finalist_summary["display_name"],
                    finalist_summary["robustness_status"],
                    finalist_summary["failed_probes"].map(
                        lambda value: f"{value} of {len(ROBUSTNESS_COLUMNS)}"
                    ),
                    finalist_summary["mean_abs_score_drift"].map(lambda value: f"{value:.4f}"),
                    finalist_summary["max_abs_score_drift"].map(lambda value: f"{value:.4f}"),
                    finalist_summary["mean_changed_class_share"].map(lambda value: f"{value:.2%}"),
                    finalist_summary["max_changed_class_share"].map(lambda value: f"{value:.2%}"),
                ],
                "fill_color": "#F7F8FA",
                "align": "left",
            },
        )
    ]
)
robustness_status_fig.update_layout(
    title="Per finalist missing feature robustness status",
    height=310,
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
robustness_status_fig.show()

latency_fig = go.Figure(
    go.Bar(
        x=finalist_summary["display_name"],
        y=finalist_summary["latency_ms_per_1k"],
        marker_color="#3776AB",
        text=finalist_summary["latency_ms_per_1k"].map(lambda value: f"{value:.2f}"),
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f} ms per 1,000 rows<extra></extra>",
    )
)
latency_fig.update_layout(
    title="Measured finalist inference latency on Val 2023",
    xaxis_title="Model",
    yaxis_title="Median milliseconds per 1,000 rows",
    template="plotly_white",
    height=480,
)
latency_fig.show()

robustness_fig = go.Figure()
robustness_fig.add_trace(
    go.Bar(
        x=finalist_summary["display_name"],
        y=finalist_summary["mean_abs_score_drift"],
        name="Mean absolute score drift",
        marker_color="#6D5DFC",
        hovertemplate="%{x}<br>Score drift %{y:.4f}<extra></extra>",
    )
)
robustness_fig.add_trace(
    go.Bar(
        x=finalist_summary["display_name"],
        y=finalist_summary["mean_changed_class_share"],
        name="Mean changed class share",
        marker_color="#F37626",
        hovertemplate="%{x}<br>Changed class share %{y:.2%}<extra></extra>",
    )
)
robustness_fig.update_layout(
    title="Measured response to missing weather and road context features",
    xaxis_title="Model",
    yaxis_title="Mean change across five probes",
    yaxis_tickformat=".1%",
    barmode="group",
    template="plotly_white",
    height=520,
)
robustness_fig.show()

# %%
finalist_predictions = {model: candidate_analysis.predictions[model] for model in finalist_models}
disagreement_long = prediction_disagreement(finalist_predictions)
disagreement_matrix = pd.DataFrame(
    0.0,
    index=finalist_models,
    columns=finalist_models,
)
for row in disagreement_long.itertuples(index=False):
    disagreement_matrix.loc[row.model_a, row.model_b] = row.disagreement_share
    disagreement_matrix.loc[row.model_b, row.model_a] = row.disagreement_share

finalist_display_names = [MODEL_LABELS[model] for model in finalist_models]
disagreement_fig = go.Figure(
    go.Heatmap(
        z=disagreement_matrix.to_numpy(),
        x=finalist_display_names,
        y=finalist_display_names,
        colorscale="Blues",
        zmin=0,
        zmax=max(0.01, float(disagreement_matrix.to_numpy().max())),
        text=np.vectorize(lambda value: f"{value:.1%}")(disagreement_matrix.to_numpy()),
        texttemplate="%{text}",
        hovertemplate="%{y} compared with %{x}<br>Disagreement %{z:.2%}<extra></extra>",
        colorbar={"title": "Share"},
    )
)
disagreement_fig.update_layout(
    title="Pairwise finalist prediction disagreement on Val 2023",
    xaxis_title="Model",
    yaxis_title="Model",
    template="plotly_white",
    height=560,
)
disagreement_fig.show()

champion_disagreement_rows = disagreement_long[
    disagreement_long["model_a"].eq(champion_artifact.model)
    | disagreement_long["model_b"].eq(champion_artifact.model)
]
champion_mean_disagreement = float(champion_disagreement_rows["disagreement_share"].mean())
maximum_disagreement_row = disagreement_long.loc[disagreement_long["disagreement_share"].idxmax()]


# %%
def write_csv_atomically(frame, path):
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomically(payload, path):
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


permutation_importance_path = PROCESSED_DIR / "c_phase_permutation_importance.csv"
analysis_manifest_path = PROCESSED_DIR / "c_phase_analysis_manifest.json"
with analysis_manifest_path.open(encoding="utf-8") as file:
    analysis_manifest = json.load(file)
assert analysis_manifest["fingerprint"] == candidate_analysis_fingerprint

permutation_cache_metadata = analysis_manifest.get(
    "permutation_importance_cache",
    {},
)
permutation_cache_hit = (
    permutation_importance_path.is_file()
    and permutation_cache_metadata.get("fingerprint") == candidate_analysis_fingerprint
    and permutation_cache_metadata.get("sample_size") == PERMUTATION_SAMPLE_SIZE
    and permutation_cache_metadata.get("repeats") == PERMUTATION_REPEATS
)

if permutation_cache_hit:
    try:
        permutation_importance_df = pd.read_csv(permutation_importance_path)
        required_importance_columns = {
            "model",
            "feature",
            "importance_mean",
            "importance_std",
            "rank",
        }
        assert required_importance_columns.issubset(permutation_importance_df.columns)
        assert set(permutation_importance_df["model"]) == set(finalist_models)
    except (AssertionError, OSError, ValueError, pd.errors.ParserError):
        permutation_cache_hit = False

if not permutation_cache_hit:
    permutation_importance_df = compute_finalist_permutation_importance(
        finalist_artifacts,
        X_val=X_validation,
        y_val=y_validation,
        sample_size=PERMUTATION_SAMPLE_SIZE,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
    )
    write_csv_atomically(
        permutation_importance_df,
        permutation_importance_path,
    )
    analysis_manifest["permutation_importance_cache"] = {
        "fingerprint": candidate_analysis_fingerprint,
        "sample_size": PERMUTATION_SAMPLE_SIZE,
        "repeats": PERMUTATION_REPEATS,
        "random_state": RANDOM_STATE,
        "models": finalist_models,
    }
    write_json_atomically(analysis_manifest, analysis_manifest_path)

print("Permutation importance cache: " + ("reused" if permutation_cache_hit else "rebuilt"))

all_importance_rank_matrix = permutation_importance_df.pivot(
    index="feature",
    columns="model",
    values="rank",
).reindex(columns=finalist_models)
champion_top_features = set(
    permutation_importance_df[
        permutation_importance_df["model"].eq(champion_artifact.model)
        & permutation_importance_df["rank"].le(10)
    ]["feature"]
)
importance_rank_correlations = {}
importance_top_10_jaccard = {}
for finalist_model in finalist_models:
    if finalist_model == champion_artifact.model:
        continue
    rank_correlation = all_importance_rank_matrix[champion_artifact.model].corr(
        all_importance_rank_matrix[finalist_model],
        method="spearman",
    )
    finalist_top_features = set(
        permutation_importance_df[
            permutation_importance_df["model"].eq(finalist_model)
            & permutation_importance_df["rank"].le(10)
        ]["feature"]
    )
    feature_union = champion_top_features | finalist_top_features
    importance_rank_correlations[finalist_model] = float(rank_correlation)
    importance_top_10_jaccard[finalist_model] = float(
        len(champion_top_features & finalist_top_features) / len(feature_union)
    )
mean_importance_rank_correlation = float(np.mean(list(importance_rank_correlations.values())))
mean_importance_top_10_jaccard = float(np.mean(list(importance_top_10_jaccard.values())))

top_feature_union = (
    permutation_importance_df[permutation_importance_df["rank"].le(10)]
    .groupby("feature")["importance_mean"]
    .max()
    .sort_values(ascending=False)
    .head(15)
    .index
)
importance_rank_matrix = (
    permutation_importance_df[permutation_importance_df["feature"].isin(top_feature_union)]
    .pivot(index="feature", columns="model", values="rank")
    .reindex(index=top_feature_union, columns=finalist_models)
)

importance_rank_fig = go.Figure(
    go.Heatmap(
        z=importance_rank_matrix.to_numpy(),
        x=finalist_display_names,
        y=importance_rank_matrix.index,
        colorscale="Blues_r",
        zmin=1,
        zmax=float(np.nanmax(importance_rank_matrix.to_numpy())),
        text=importance_rank_matrix.to_numpy(),
        texttemplate="%{text:.0f}",
        hovertemplate="%{y}<br>%{x}<br>Rank %{z:.0f}<extra></extra>",
        colorbar={"title": "Rank"},
    )
)
importance_rank_fig.update_layout(
    title="Val 2023 permutation importance rank by finalist",
    xaxis_title="Model",
    yaxis_title="Input feature",
    template="plotly_white",
    height=650,
)
importance_rank_fig.show()

# %%
qualitative_rows = [
    {
        "model": row.display_name,
        "macro_f1": row.macro_f1,
        "recall_ksi": row.recall_ksi,
        "latency_ms_per_1k": row.latency_ms_per_1k,
        "robustness_score": row.robustness_score,
    }
    for row in finalist_summary.itertuples(index=False)
]
qualitative_matrix = build_qualitative_matrix(qualitative_rows)
criteria_used = qualitative_matrix.attrs["criteria_used"]
criteria_excluded = qualitative_matrix.attrs["criteria_excluded"]

qualitative_fig = go.Figure(
    data=[
        go.Table(
            header={
                "values": [
                    "Model",
                    "Macro F1",
                    "Recall KSI",
                    "Latency ms per 1,000 rows",
                    "Robustness score",
                    "Measured decision score",
                ],
                "fill_color": "#14213D",
                "font": {"color": "white"},
                "align": "left",
            },
            cells={
                "values": [
                    qualitative_matrix["model"],
                    qualitative_matrix["macro_f1"].map(lambda value: f"{value:.3f}"),
                    qualitative_matrix["recall_ksi"].map(lambda value: f"{value:.3f}"),
                    qualitative_matrix["latency_ms_per_1k"].map(lambda value: f"{value:.2f}"),
                    qualitative_matrix["robustness_score"].map(lambda value: f"{value:.3f}"),
                    qualitative_matrix["weighted_score"].map(lambda value: f"{value:.3f}"),
                ],
                "fill_color": "#F7F8FA",
                "align": "left",
            },
        )
    ]
)
qualitative_fig.update_layout(
    title="Finalist decision matrix using only measured and varying criteria",
    height=310,
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
qualitative_fig.show()

print(f"Criteria used: {', '.join(criteria_used)}")
print(f"Criteria excluded: {criteria_excluded}")

# %% [markdown]
# The decision matrix excludes any missing or constant criterion automatically. It does not assign subjective interpretability or training cost scores. The disagreement and importance views remain diagnostic evidence because they describe model diversity and feature dependence, not a universally better direction.

# %% [markdown]
# ## 4 Champion only Test 2024 confirmation and gate
#
# The candidate comparison now stops. Only the Random Forest champion is loaded from the dedicated deployment artifact. Its threshold was selected on Val 2023. Test 2024 confirms the frozen model and threshold once, with no challenger predictions and no test guided reselection.

# %%
champion_pipeline = joblib.load(PROCESSED_DIR / "a3_binary_best_model.joblib")
champion_threshold = float(model_card["optimal_threshold_val_2023"])
champion_test_scores = champion_pipeline.predict_proba(X_test)[:, 1]
champion_test_predictions = (champion_test_scores >= champion_threshold).astype(int)
champion_test_metrics = evaluate_binary_predictions(
    y_test.to_numpy(),
    champion_test_predictions,
)

recorded_test_metrics = model_card["test_2024_metrics"]
CACHE_TOLERANCE_RELATIVE = 0.01
for metric_name in ["macro_f1", "recall_ksi", "recall_slight"]:
    observed = float(champion_test_metrics[metric_name])
    recorded = float(recorded_test_metrics[metric_name])
    denominator = max(abs(recorded), 1e-12)
    relative_drift = abs(observed - recorded) / denominator
    assert relative_drift < CACHE_TOLERANCE_RELATIVE, (
        f"{metric_name} drift {relative_drift:.2%} exceeds the "
        f"{CACHE_TOLERANCE_RELATIVE:.0%} cache tolerance."
    )

test_confusion = confusion_matrix(
    y_test,
    champion_test_predictions,
    labels=[1, 0],
)
test_confusion_fig = plot_confusion_matrix_heatmap(
    test_confusion,
    labels=["KSI", "Slight injury"],
    title="Random Forest confusion matrix on Test 2024",
)
test_confusion_fig.update_layout(height=560)
test_confusion_fig.show()

gate_rows = pd.DataFrame(
    [
        {
            "criterion": "Macro F1 at least 0.55",
            "value": champion_test_metrics["macro_f1"],
            "passed": champion_test_metrics["macro_f1"] >= 0.55,
        },
        {
            "criterion": "Recall KSI at least 0.50",
            "value": champion_test_metrics["recall_ksi"],
            "passed": champion_test_metrics["recall_ksi"] >= 0.50,
        },
    ]
)
gate_passed = bool(gate_rows["passed"].all())
gate_fig = go.Figure(
    data=[
        go.Table(
            header={
                "values": ["Test 2024 criterion", "Observed value", "Passed"],
                "fill_color": "#14213D",
                "font": {"color": "white"},
                "align": "left",
            },
            cells={
                "values": [
                    gate_rows["criterion"],
                    gate_rows["value"].map(lambda value: f"{value:.4f}"),
                    gate_rows["passed"].map({True: "Yes", False: "No"}),
                ],
                "fill_color": "#F7F8FA",
                "align": "left",
            },
        )
    ]
)
gate_fig.update_layout(
    title="Frozen champion gate result",
    height=250,
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
gate_fig.show()

# %% [markdown]
# The confirmation checks the three recorded scalar metrics against a one percent relative cache tolerance. This allows small feature cache refresh differences while still stopping the notebook if the persisted model and recorded result no longer agree.

# %% [markdown]
# ## 5 Champion error analysis
#
# The confusion matrix gives the overall error count. Slice analysis then asks where the champion misses KSI cases or flags slight injury cases as KSI. Rates use the relevant actual class as denominator, so slices with different class balance remain comparable.

# %%
LIGHT_LABELS = {0: "Daylight", 1: "Twilight", 2: "Darkness"}
ROAD_CONDITION_LABELS = {
    0: "Dry",
    1: "Wet or slippery",
    2: "Winter conditions",
}
ACCIDENT_TYPE_LABELS = {
    0: "Other accident type",
    1: "Collision with a parked or stopped vehicle",
    2: "Rear end collision",
    3: "Side collision with a vehicle moving in the same direction",
    4: "Collision with oncoming traffic",
    5: "Crossing or turning collision",
    6: "Collision with a pedestrian",
    7: "Collision with a road obstacle",
    8: "Departure from road to the right",
    9: "Departure from road to the left",
}


def decode_slice_label_english(column, value):
    if column == "UART":
        return f"Accident type: {ACCIDENT_TYPE_LABELS.get(int(value), value)}"
    if column == "STRZUSTAND":
        return f"Road condition: {ROAD_CONDITION_LABELS.get(int(value), value)}"
    if column == "ULICHTVERH":
        return f"Light condition: {LIGHT_LABELS.get(int(value), value)}"
    if column == "USTUNDE":
        return f"Hour: {int(value):02d}:00"
    if column == "osm_dominant_road_class":
        return f"OSM road class: {value}"
    if column == "_precip_bucket":
        return f"Precipitation: {value}"
    return f"{column}: {value}"


slice_columns = [
    "UART",
    "osm_dominant_road_class",
    "STRZUSTAND",
    "ULICHTVERH",
    "_precip_bucket",
    "USTUNDE",
]
slice_frame = test_frame[slice_columns].reset_index(drop=True)
error_slices = compute_error_slices(
    pd.Series(y_test.to_numpy()),
    pd.Series(champion_test_predictions),
    slice_frame,
    slice_columns,
)
error_slices["label"] = [
    decode_slice_label_english(column, value)
    for column, value in zip(
        error_slices["slice_column"],
        error_slices["slice_value"],
        strict=True,
    )
]

error_plot_data = (
    error_slices[error_slices["n"].ge(100)]
    .nlargest(15, "false_negative_rate")
    .sort_values("false_negative_rate")
)
error_slice_fig = go.Figure(
    go.Bar(
        x=error_plot_data["false_negative_rate"],
        y=error_plot_data["label"],
        orientation="h",
        marker_color="#3776AB",
        text=error_plot_data["false_negative_rate"].map(lambda value: f"{value:.1%}"),
        textposition="outside",
        customdata=error_plot_data[["n", "n_false_negative"]],
        hovertemplate=(
            "%{y}<br>False negative rate %{x:.1%}"
            "<br>Rows %{customdata[0]:,.0f}"
            "<br>False negatives %{customdata[1]:,.0f}<extra></extra>"
        ),
    )
)
error_slice_fig.update_layout(
    title="Highest champion false negative rates by Test 2024 slice",
    xaxis_title="False negative rate among actual KSI cases",
    xaxis_tickformat=".0%",
    yaxis_title="Slice",
    template="plotly_white",
    height=680,
    margin={"l": 320, "r": 70, "t": 60, "b": 60},
)
error_slice_fig.show()

# %% [markdown]
# The slice chart is descriptive rather than causal. A high rate can indicate weak signal, a difficult subgroup, or a small effective KSI denominator. The row count and false negative count remain available in the hover details.

# %% [markdown]
# ## 6 Cross model feature evidence and champion SHAP
#
# Permutation importance in Section 3 is model agnostic and compares all four finalists on the same Val 2023 sample. The SHAP analysis below answers a different question. It explains how the Random Forest champion uses its transformed features on a stratified Test 2024 sample.
#
# The champion contains very deep trees. Exact Tree SHAP was not practical in the earlier timing check, so this notebook uses the documented approximate TreeExplainer path with additivity checking disabled. The sample contains 2,500 KSI cases and 2,500 slight injury cases.

# %%
FEATURE_LABELS_ENGLISH = {
    "IstRad": "Bicycle involved",
    "IstPKW": "Car involved",
    "IstFuss": "Pedestrian involved",
    "IstKrad": "Motorcycle involved",
    "IstGkfz": "Goods vehicle involved",
    "IstSonstig": "Other transport involved",
    "LON": "Longitude",
    "LAT": "Latitude",
    "dwd_temp_air_2m": "Air temperature",
    "dwd_precip_mm": "Precipitation",
    "dwd_visibility_m": "Visibility",
    "dwd_wind_speed_ms": "Wind speed",
    "dwd_station_dist_km": "Distance to weather station",
    "osm_road_density": "OSM road density",
    "osm_way_count": "OSM way count",
    "osm_maxspeed_mean": "OSM mean speed limit",
    "osm_maxspeed_max": "OSM maximum speed limit",
}
ACCIDENT_CATEGORY_LABELS = {
    1: "Driving accident",
    2: "Turning accident",
    3: "Crossing accident",
    4: "Pedestrian crossing accident",
    5: "Stationary traffic accident",
    6: "Longitudinal traffic accident",
    7: "Other accident",
}


def humanize_feature_name_english(name):
    for column, labels in {
        "UART": ACCIDENT_TYPE_LABELS,
        "UTYP1": ACCIDENT_CATEGORY_LABELS,
        "ULICHTVERH": LIGHT_LABELS,
        "STRZUSTAND": ROAD_CONDITION_LABELS,
    }.items():
        prefix = f"{column}_"
        if name.startswith(prefix):
            value = name[len(prefix) :]
            return labels.get(int(value), value)
    road_prefix = "osm_dominant_road_class_"
    if name.startswith(road_prefix):
        return f"OSM road class: {name[len(road_prefix) :]}"
    if name.endswith("_target_enc"):
        base = name.removesuffix("_target_enc")
        return f"{base} target encoding"
    if name.endswith(("_sin", "_cos")):
        base, component = name.rsplit("_", 1)
        return f"{base} cyclic {component}"
    return FEATURE_LABELS_ENGLISH.get(name, name)


test_target_series = pd.Series(y_test.to_numpy())
shap_indices = (
    test_target_series.groupby(test_target_series).sample(n=2_500, random_state=RANDOM_STATE).index
)
shap_raw = X_test.iloc[shap_indices].reset_index(drop=True)
shap_target = y_test.iloc[shap_indices].reset_index(drop=True)

champion_preprocessor = champion_pipeline[:-1]
champion_classifier = champion_pipeline[-1]
shap_features = pd.DataFrame(
    champion_preprocessor.transform(shap_raw),
    columns=champion_preprocessor.get_feature_names_out(),
).astype(float)

shap_explainer = shap.TreeExplainer(champion_classifier)
shap_values = shap_explainer.shap_values(
    shap_features,
    approximate=True,
    check_additivity=False,
)
shap_values_ksi = shap_values[1] if isinstance(shap_values, list) else shap_values
if shap_values_ksi.ndim == 3:
    shap_values_ksi = shap_values_ksi[:, :, 1]

assert shap_values_ksi.shape == shap_features.shape
shap_importance = pd.Series(
    np.abs(shap_values_ksi).mean(axis=0),
    index=shap_features.columns,
).sort_values(ascending=False)

# %%
beeswarm_feature_count = 20
beeswarm_features = shap_importance.head(beeswarm_feature_count).index[::-1]
feature_positions = {feature: index for index, feature in enumerate(shap_features.columns)}
jitter_rng = np.random.default_rng(RANDOM_STATE)

beeswarm_fig = go.Figure()
for row_index, feature in enumerate(beeswarm_features):
    feature_values = shap_features[feature].to_numpy()
    feature_shap = shap_values_ksi[:, feature_positions[feature]]
    value_span = feature_values.max() - feature_values.min()
    normalized_values = (feature_values - feature_values.min()) / (
        value_span if value_span else 1.0
    )
    jitter = jitter_rng.uniform(-0.35, 0.35, size=len(feature_shap))
    beeswarm_fig.add_trace(
        go.Scatter(
            x=feature_shap,
            y=row_index + jitter,
            mode="markers",
            marker={
                "color": normalized_values,
                "colorscale": "RdBu_r",
                "size": 4,
                "opacity": 0.65,
                "showscale": row_index == 0,
                "colorbar": {
                    "title": "Feature value",
                    "tickvals": [0, 1],
                    "ticktext": ["Low", "High"],
                    "x": 1.02,
                }
                if row_index == 0
                else None,
            },
            name=humanize_feature_name_english(feature),
            hovertemplate=(
                f"{humanize_feature_name_english(feature)}<br>SHAP value %{{x:.3f}}<extra></extra>"
            ),
            showlegend=False,
        )
    )
beeswarm_fig.update_layout(
    title="Random Forest SHAP beeswarm on the stratified Test 2024 sample",
    xaxis_title="SHAP contribution toward the KSI score",
    yaxis={
        "tickmode": "array",
        "tickvals": list(range(beeswarm_feature_count)),
        "ticktext": [humanize_feature_name_english(feature) for feature in beeswarm_features],
    },
    template="plotly_white",
    height=760,
    margin={"l": 240, "r": 130, "t": 60, "b": 60},
)
beeswarm_fig.show()

# %%
bar_feature_count = 15
bar_data = shap_importance.head(bar_feature_count).iloc[::-1]
shap_bar_fig = go.Figure(
    go.Bar(
        x=bar_data.to_numpy(),
        y=[humanize_feature_name_english(feature) for feature in bar_data.index],
        orientation="h",
        marker_color="#3776AB",
        hovertemplate="%{y}<br>Mean absolute SHAP %{x:.4f}<extra></extra>",
    )
)
shap_bar_fig.update_layout(
    title="Random Forest mean absolute SHAP values",
    xaxis_title="Mean absolute SHAP value",
    yaxis_title="Transformed feature",
    template="plotly_white",
    height=560,
    margin={"l": 260, "r": 40, "t": 60, "b": 60},
)
shap_bar_fig.show()

# %%
sample_scores = champion_pipeline.predict_proba(shap_raw)[:, 1]
sample_predictions = (sample_scores >= champion_threshold).astype(int)

case_masks = {
    "True positive KSI": ((shap_target.to_numpy() == 1) & (sample_predictions == 1)),
    "False negative KSI": ((shap_target.to_numpy() == 1) & (sample_predictions == 0)),
    "False positive slight injury": ((shap_target.to_numpy() == 0) & (sample_predictions == 1)),
    "True negative slight injury": ((shap_target.to_numpy() == 0) & (sample_predictions == 0)),
}
case_indices = {
    case_name: int(np.flatnonzero(mask)[0])
    for case_name, mask in case_masks.items()
    if np.flatnonzero(mask).size
}
assert len(case_indices) == 4

expected_value = np.atleast_1d(shap_explainer.expected_value)
expected_value_ksi = float(expected_value[1] if len(expected_value) > 1 else expected_value[0])

waterfall_feature_count = 10
for case_name, case_index in case_indices.items():
    contributions = pd.Series(
        shap_values_ksi[case_index],
        index=shap_features.columns,
    )
    top_contributions = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    ).head(waterfall_feature_count)
    remaining_contribution = contributions.drop(top_contributions.index).sum()

    waterfall_labels = [
        "Base value",
        *(humanize_feature_name_english(feature) for feature in top_contributions.index),
        "Remaining features",
        "Prediction",
    ]
    waterfall_values = [
        expected_value_ksi,
        *top_contributions.to_list(),
        remaining_contribution,
        0,
    ]
    waterfall_measures = [
        "absolute",
        *(["relative"] * (len(top_contributions) + 1)),
        "total",
    ]

    waterfall_fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=waterfall_measures,
            x=waterfall_labels,
            y=waterfall_values,
            connector={"line": {"color": "rgba(80, 80, 80, 0.4)"}},
            increasing={"marker": {"color": "#E63946"}},
            decreasing={"marker": {"color": "#3776AB"}},
            totals={"marker": {"color": "#14213D"}},
        )
    )
    waterfall_fig.update_layout(
        title=f"Random Forest SHAP case: {case_name}",
        yaxis_title="Contribution toward the KSI score",
        template="plotly_white",
        height=520,
        xaxis_tickangle=-35,
    )
    waterfall_fig.show()

# %% [markdown]
# The permutation ranks and SHAP values should not be read as causal effects. Permutation importance measures validation performance loss when a raw input is disrupted. SHAP distributes one champion prediction across transformed features. Agreement between the two views strengthens a descriptive interpretation, while disagreement can reveal model specific feature use.

# %% [markdown]
# ## 7 Literature context and limitations
#
# The result is consistent with the literature context established in the Q and A³ phases. Public accident records contain useful road, time, weather, location, and participant signals, but they omit several physical determinants of injury severity. The binary KSI target is therefore more feasible than the original three class target without becoming an easy prediction problem.
#
# The main limitations are:
#
# 1. The Unfallatlas covers police recorded accidents. Unreported accidents are outside the observed population.
# 2. Impact speed, restraint use, occupant age, vehicle mass, and detailed injury mechanisms are not available in the public feature set.
# 3. OSM road context is not historically versioned for every accident year.
# 4. Training covers 2016-2022, validation uses 2023, and the final confirmation uses 2024. Later years can still drift.
# 5. Missing feature probes test operational resilience, not every possible distribution shift.
# 6. Permutation importance and SHAP describe association and model behavior, not causal effects.
# 7. The selected threshold reflects the stated macro F1 and Recall KSI gate. A different operational cost function could justify a different threshold.

# %% [markdown]
# ## 8 Final model decision and K phase contract
#
# The measured matrix ranks the finalists using macro F1, Recall KSI, latency, and robustness only when those criteria are present and varying. Pairwise disagreement and permutation rank agreement are diagnostic evidence without a universally better direction. The final statement reports their measured values, but it does not turn them into arbitrary ranking points.
#
# The matrix can confirm the preselected champion or reveal a validation challenger. It cannot silently replace the champion after Test 2024 has been opened. If another finalist leads the measured matrix, it becomes a candidate for a future preregistered comparison while Random Forest remains the deployment model for this cycle.

# %%
measured_matrix_leader = str(qualitative_matrix.iloc[0]["model"])
champion_display_name = MODEL_LABELS[champion_artifact.model]
random_forest_remains_preferred = measured_matrix_leader == champion_display_name

fastest_row = finalist_summary.loc[finalist_summary["latency_ms_per_1k"].idxmin()]
most_stable_row = finalist_summary.sort_values(
    ["failed_probes", "mean_changed_class_share", "mean_abs_score_drift"],
    ascending=[True, True, True],
).iloc[0]

if random_forest_remains_preferred:
    decision_statement = (
        "Random Forest remains preferred after measured performance, latency, "
        "and robustness are ranked together. Disagreement and permutation "
        "rank evidence describe model diversity but do not reverse that result."
    )
else:
    decision_statement = (
        f"{measured_matrix_leader} leads the measured validation matrix. "
        "Random Forest is therefore not the preferred finalist on the combined "
        "measured criteria, but it remains the deployment champion because the "
        "test set cannot be reused for challenger selection."
    )

finalist_measurement_records = []
for row in finalist_summary.itertuples(index=False):
    finalist_measurement_records.append(
        {
            "model": row.display_name,
            "registry_model": row.model,
            "macro_f1": float(row.macro_f1),
            "recall_ksi": float(row.recall_ksi),
            "recall_slight": float(row.recall_slight),
            "latency_ms_per_1k": float(row.latency_ms_per_1k),
            "robustness_status": row.robustness_status,
            "failed_probes": int(row.failed_probes),
            "mean_abs_score_drift": float(row.mean_abs_score_drift),
            "max_abs_score_drift": float(row.max_abs_score_drift),
            "mean_changed_class_share": float(row.mean_changed_class_share),
            "max_changed_class_share": float(row.max_changed_class_share),
            "robustness_score": float(row.robustness_score),
        }
    )

disagreement_records = [
    {
        "model_a": MODEL_LABELS[row.model_a],
        "model_b": MODEL_LABELS[row.model_b],
        "disagreement_share": float(row.disagreement_share),
    }
    for row in disagreement_long.itertuples(index=False)
]

inference_contract["decision_evidence"] = {
    "candidate_registry_size": len(candidate_artifacts),
    "validation_decision_matrix": {
        "leader": measured_matrix_leader,
        "criteria_used": criteria_used,
        "criteria_excluded": criteria_excluded,
        "ranking": [
            {
                "model": row.model,
                "measured_decision_score": float(row.weighted_score),
            }
            for row in qualitative_matrix.itertuples(index=False)
        ],
    },
    "finalist_measurements": finalist_measurement_records,
    "latency_summary": {
        "fastest_model": str(fastest_row["display_name"]),
        "fastest_ms_per_1k": float(fastest_row["latency_ms_per_1k"]),
    },
    "robustness_summary": {
        "most_stable_model": str(most_stable_row["display_name"]),
        "failed_probes": int(most_stable_row["failed_probes"]),
        "mean_abs_score_drift": float(most_stable_row["mean_abs_score_drift"]),
        "mean_changed_class_share": float(most_stable_row["mean_changed_class_share"]),
    },
    "pairwise_disagreement": {
        "champion_mean_disagreement": champion_mean_disagreement,
        "maximum_pair": {
            "model_a": MODEL_LABELS[maximum_disagreement_row["model_a"]],
            "model_b": MODEL_LABELS[maximum_disagreement_row["model_b"]],
            "disagreement_share": float(maximum_disagreement_row["disagreement_share"]),
        },
        "pairs": disagreement_records,
    },
    "permutation_rank_evidence": {
        "fingerprint": candidate_analysis_fingerprint,
        "sample_size": PERMUTATION_SAMPLE_SIZE,
        "repeats": PERMUTATION_REPEATS,
        "mean_spearman_correlation_with_champion": (mean_importance_rank_correlation),
        "mean_top_10_jaccard_with_champion": (mean_importance_top_10_jaccard),
        "spearman_correlation_by_model": {
            MODEL_LABELS[model]: value for model, value in importance_rank_correlations.items()
        },
        "top_10_jaccard_by_model": {
            MODEL_LABELS[model]: value for model, value in importance_top_10_jaccard.items()
        },
    },
    "preference_conclusion": {
        "random_forest_remains_preferred": (random_forest_remains_preferred),
        "statement": decision_statement,
        "diagnostic_evidence_has_no_preferred_direction": True,
    },
    "deployment_model": champion_display_name,
    "deployment_model_registry_name": champion_artifact.model,
    "deployment_model_sha256": champion_artifact.sha256,
    "test_2024_champion_only": True,
    "test_2024_metrics": {
        key: value for key, value in champion_test_metrics.items() if key != "confusion_matrix"
    },
    "acceptance_gate_passed": gate_passed,
    "analysis_artifacts": {
        "manifest": "data/processed/c_phase_analysis_manifest.json",
        "candidate_metrics": "data/processed/c_phase_candidate_metrics.csv",
        "candidate_scores": "data/processed/c_phase_candidate_scores.parquet",
        "candidate_robustness": "data/processed/c_phase_candidate_robustness.csv",
        "permutation_importance": "data/processed/c_phase_permutation_importance.csv",
    },
}

contract_path = PROCESSED_DIR / "c_phase_inference_contract.json"
write_json_atomically(inference_contract, contract_path)

decision_fig = go.Figure(
    data=[
        go.Table(
            header={
                "values": ["Decision item", "Measured result"],
                "fill_color": "#14213D",
                "font": {"color": "white"},
                "align": "left",
            },
            cells={
                "values": [
                    [
                        "Validation matrix leader",
                        "Fastest finalist",
                        "Most stable missing feature response",
                        "Random Forest mean disagreement",
                        "Mean importance rank correlation with Random Forest",
                        "Random Forest remains preferred",
                        "Deployment model",
                        "Test 2024 gate",
                    ],
                    [
                        measured_matrix_leader,
                        (
                            f"{fastest_row['display_name']} at "
                            f"{fastest_row['latency_ms_per_1k']:.2f} ms per 1,000 rows"
                        ),
                        (
                            f"{most_stable_row['display_name']} with "
                            f"{int(most_stable_row['failed_probes'])} failed probes and "
                            f"{most_stable_row['mean_changed_class_share']:.2%} mean class change"
                        ),
                        f"{champion_mean_disagreement:.2%}",
                        f"{mean_importance_rank_correlation:.3f}",
                        "Yes" if random_forest_remains_preferred else "No",
                        champion_display_name,
                        "Passed" if gate_passed else "Not passed",
                    ],
                ],
                "fill_color": "#F7F8FA",
                "align": "left",
            },
        )
    ]
)
decision_fig.update_layout(
    title="Final model decision and K phase handoff",
    height=420,
    margin={"l": 20, "r": 20, "t": 55, "b": 20},
)
decision_fig.show()
print(decision_statement)
print(f"Inference contract written to {contract_path}")

# %% [markdown]
# ## 9 C phase summary
#
# The C phase now uses the complete persisted model set instead of a champion only approximation. All ten candidates are validated on Val 2023, the four finalists receive independent latency and robustness measurements, prediction disagreement exposes where their decisions differ, and permutation importance compares their raw feature dependence.
#
# Test 2024 remains a single confirmation of the frozen Random Forest champion. Champion error slices and SHAP then explain the confirmed model without leaking challenger information into the final decision.
#
# The generated analysis files and inference contract provide the K phase with a traceable model path, threshold, input schema, registry fingerprint, validation evidence, and final gate result.
