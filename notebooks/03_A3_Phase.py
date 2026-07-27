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
#     display_name: unfallatlas-qua3ck (3.13.11)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Unfallatlas Deutschland: A³ Phase
#
# **Phase:** Algorithm / Adapt / Adjust (A³), 3 of 5 in QUA³CK
#
# **Goal.** Test whether the original three-class severity target is feasible,
# document the evidence for the decision, then build and validate a purpose-built
# binary KSI model.
#
# **Scope.** This phase covers model search, imbalance handling, tuning,
# threshold selection, and one final Test-2024 evaluation. Phase C continues
# with cross-model comparison, explainability, limitations, and the final
# recommendation.

# %% [markdown]
# ## Position in the QUA³CK process
#
# | Phase | Notebook | Status |
# |:---|:---|:---:|
# | Q: Question | `01_Q_Phase.ipynb` | Complete |
# | U: Understanding | `02_U_Phase.ipynb` | Complete |
# | **A³: Algorithm / Adapt / Adjust** | `03_A3_Phase.ipynb` | **Current phase** |
# | C: Conclude & Compare | `04_C_Phase.ipynb` | Next |
# | K: Knowledge Transfer | `app/streamlit_app.py` | Planned |

# %% [markdown]
# ## 0 Setup and reproducibility

# %%
import json
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.io as pio

from unfallatlas.features.preprocessing import (
    chronological_split,
    load_training_frame,
    split_features_target,
    split_features_target_binary,
)
from unfallatlas.models.artifacts import validate_candidate_registry
from unfallatlas.models.evaluate import (
    evaluate_binary_predictions,
    evaluate_predictions,
    meets_acceptance_criteria,
    meets_binary_acceptance_criteria,
    select_best_candidate,
)
from unfallatlas.models.imbalance import find_gate_optimal_offsets
from unfallatlas.viz.metrics_viz import (
    plot_binary_f1_recall_front,
    plot_confusion_matrix_heatmap,
    plot_f1_recall_front,
)

# %%
pio.templates.default = "plotly_white"
pio.renderers.default = "vscode"
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
np.random.seed(42)

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG_DIR = BASE_DIR / "reports" / "figures" / "a3_phase"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _git_short_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# %% [markdown]
# ### Persisted-evidence execution mode
#
# This presentation is audit-only. Every fitted model, comparison table, model
# card, and Optuna study comes from scientific run `b1ea31e`.
#
# Notebook execution never trains a model or schedules a new trial. Missing or
# invalid evidence raises an error instead of silently starting expensive work.

# %%
AUDIT_ONLY = True
MODEL_RUN_CHECKPOINT_ID = "b1ea31e"
print(f"Execution mode: {'audit-only' if AUDIT_ONLY else 'training'}")
print(f"Scientific model run: {MODEL_RUN_CHECKPOINT_ID}")

# %% [markdown]
# ### Evidence paths and provenance
#
# The fixed checkpoint identifier separates the scientific run from later
# editorial commits. This preserves reproducibility and keeps presentation
# execution deterministic.

# %%
PROGRESS_LOG = BASE_DIR / "reports" / "a3_progress.log"
CHECKPOINT_DIR = PROCESSED_DIR / "a3_checkpoints" / MODEL_RUN_CHECKPOINT_ID
OPTUNA_DB_PATH = CHECKPOINT_DIR / "optuna_study.db"

required_evidence = [
    CHECKPOINT_DIR,
    OPTUNA_DB_PATH,
    PROCESSED_DIR / "a3_model_comparison.csv",
    PROCESSED_DIR / "a3_model_card.json",
    PROCESSED_DIR / "a3_best_model.joblib",
    PROCESSED_DIR / "a3_binary_model_comparison.csv",
    PROCESSED_DIR / "a3_binary_model_card.json",
    PROCESSED_DIR / "a3_binary_best_model.joblib",
]
missing_evidence = [item for item in required_evidence if not item.exists()]
if missing_evidence:
    raise FileNotFoundError(f"Audit-only execution requires persisted evidence: {missing_evidence}")

print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Optuna database: {OPTUNA_DB_PATH.name}")

# %% [markdown]
# ## 1 Data contract and chronological evaluation protocol
#
# A³ reads the enriched feature cache produced in the U phase. The chronological
# split remains fixed throughout the phase:
#
# - Train: 2016-2022
# - Validation: 2023
# - Test: 2024
#
# Model families, imbalance strategies, hyperparameters, and decision thresholds
# are selected without inspecting Test-2024. The test set is reserved for one
# final evaluation of the selected binary pipeline.

# %%
df = load_training_frame(BASE_DIR)
train_df, val_df, test_df = chronological_split(df)

X_train, y_train = split_features_target(train_df)
X_val, y_val = split_features_target(val_df)

provenance = {
    "rows_train": len(X_train),
    "rows_val": len(X_val),
    "rows_test_reserved": len(test_df),
    "git_commit": _git_short_sha(),
    "model_run_checkpoint_id": MODEL_RUN_CHECKPOINT_ID,
    "audited_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
}
for key, value in provenance.items():
    print(f"  {key:24s} {value}")

# %% [markdown]
# ### Year-grouped cross-validation
#
# `GroupKFold` keeps calendar years intact during tuning. This provides a
# time-aware validation signal without mixing observations from the same year
# across folds.
#
# The following cell is a protocol check. Later tuning cells construct their own
# year groups from the relevant training subsample.

# %%
training_years = sorted(train_df["UJAHR"].unique().tolist())
print(f"Training years: {training_years}")
print("Persisted studies use year-grouped cross-validation.")
print("Test-2024 remains reserved until section 11.")

# %% [markdown]
# ## Part I: Three-class feasibility
#
# The first part treats the original three-class severity target as a feasibility
# question. It establishes the baseline range, tests the planned modelling
# strategies, and then asks whether the acceptance gate is reachable with the
# available public features.

# %% [markdown]
# ## 2 Baselines and candidate families
#
# Random guess and majority class establish the lower performance range.
# Logistic Regression provides the first learned reference point.

# %%
comparison_df = pd.read_csv(PROCESSED_DIR / "a3_model_comparison.csv")
required_multiclass_columns = {
    "model",
    "macro_f1",
    "recall_class_1",
}
if not required_multiclass_columns.issubset(comparison_df.columns):
    raise ValueError("The persisted three-class comparison is incomplete.")

baseline_names = {
    "random_guess",
    "majority_class",
    "logistic_regression",
}
baseline_rows = comparison_df[comparison_df["model"].isin(baseline_names)].sort_values(
    "macro_f1", ascending=False
)
print(f"Persisted three-class configurations: {len(comparison_df)}")
baseline_rows

# %% [markdown]
# ### Tree ensemble candidate families
#
# Random Forest, XGBoost, LightGBM, and CatBoost are evaluated with default and
# balanced weighting. All eight configurations are scored on Val-2023 under the
# same preprocessing and evaluation contract.

# %%
candidate_families = ["catboost", "lightgbm"]
candidate_names = {family: f"{family}_balanced" for family in candidate_families}

tree_stage_rows = comparison_df[
    comparison_df["model"].str.endswith(("_default", "_balanced"))
].sort_values("macro_f1", ascending=False)
print(f"Persisted Stage 1 tree configurations: {len(tree_stage_rows)}")
tree_stage_rows

# %% [markdown]
# ### Gate-aware family selection
#
# Candidate selection considers macro-F1 only after the Recall(class 1) gate.
# Random Forest has the highest raw macro-F1 in the initial comparison, but its
# Recall(class 1) is too low. CatBoost and LightGBM clear the recall gate and
# therefore advance to the imbalance and tuning comparison.

# %%
candidate_rows = comparison_df[comparison_df["model"].isin(candidate_names.values())]
for row in candidate_rows.itertuples():
    print(
        f"Candidate: {row.model}, macro-F1={row.macro_f1:.3f}, "
        f"Recall(class 1)={row.recall_class_1:.3f}"
    )
if set(candidate_rows["model"]) != set(candidate_names.values()):
    raise ValueError("A persisted three-class candidate row is missing.")

# %%
fig = px.bar(
    comparison_df.sort_values("macro_f1"),
    x="macro_f1",
    y="model",
    orientation="h",
    title="Validation macro-F1 by model: persisted three-class search",
)
fig.add_vline(
    x=0.55,
    line_dash="dash",
    annotation_text="Acceptance threshold (0.55)",
)
fig.write_html(
    FIG_DIR / "05_model_comparison_stage0_1.html",
    include_plotlyjs=False,
)
fig.show()


# %% [markdown]
# The first comparison narrows the search to two viable
# three-class families. The next section tests whether imbalance handling and
# tuning can move either family into the acceptance region.

# %% [markdown]
# ## 3 Imbalance strategies and tuning
#
# CatBoost and LightGBM are compared with balanced weighting, SMOTE, ADASYN,
# threshold moving, and ordinal classification on a stratified training
# subsample of at most 500,000 rows. Only scientifically valid and operationally
# supported configurations can advance to tuning.


# %%
strategy_names = {
    family: [
        f"{family}_smote",
        f"{family}_adasyn",
        f"{family}_threshold_moving",
        f"{family}_ordinal",
        candidate_names[family],
    ]
    for family in candidate_families
}
winning_strategy_per_family = {}
for family, names in strategy_names.items():
    strategy_rows = comparison_df[comparison_df["model"].isin(names)]
    if strategy_rows.empty:
        raise ValueError(f"No persisted strategy evidence for {family}.")
    winning_strategy_per_family[family] = select_best_candidate(strategy_rows)
    winner = winning_strategy_per_family[family]
    print(
        f"{family} strategy winner: {winner['model']}, "
        f"macro-F1={winner['macro_f1']:.3f}, "
        f"Recall(class 1)={winner['recall_class_1']:.3f}"
    )

# %%
strategy_plot_rows = comparison_df[
    comparison_df["model"].isin([name for names in strategy_names.values() for name in names])
].copy()
strategy_fig = px.bar(
    strategy_plot_rows.sort_values("macro_f1"),
    x="macro_f1",
    y="model",
    orientation="h",
    color=strategy_plot_rows["model"].str.split("_").str[0],
    labels={"color": "Family"},
    title="Persisted imbalance-strategy comparison",
)
strategy_fig.show()

# %% [markdown]
# ### Family-specific Optuna tuning
#
# Each viable family receives the same nine-trial budget with year-grouped
# cross-validation. Trial selection remains gate-aware. The persisted study
# counts only completed trials when deciding whether further work is required,
# so rerunning the notebook does not silently add another full budget.

# %%
optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna_storage = f"sqlite:///{OPTUNA_DB_PATH}"
tuned_candidates = {}

for family in candidate_families:
    study_name = f"a3_tuning_{family}"
    study = optuna.load_study(
        study_name=study_name,
        storage=optuna_storage,
    )
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    failed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]
    if len(completed) < 9:
        raise RuntimeError(f"{study_name} has only {len(completed)} completed trials.")

    trial_rows = pd.DataFrame(
        [
            {
                "params": trial.params,
                "macro_f1": trial.value,
                "recall_class_1": trial.user_attrs["recall_class_1"],
            }
            for trial in completed
        ]
    )
    best_trial = select_best_candidate(trial_rows)
    tuned_candidates[family] = {
        "best_params": best_trial["params"],
        "cv_macro_f1": float(best_trial["macro_f1"]),
        "cv_recall_class_1": float(best_trial["recall_class_1"]),
    }
    print(
        f"{study_name}: {len(completed)} complete, {len(failed)} failed. "
        "Persisted study audited; no trials scheduled."
    )

# %% [markdown]
# ## 4 Three-class validation result
#
# Gate-aware cross-validation selects the final three-class family. Its tuned
# pipeline is refit on the full 2016-2022 training period and evaluated on
# Val-2023 only. Test-2024 remains untouched because this formulation is still
# being assessed for feasibility.

# %%
family_comparison = pd.DataFrame(
    [
        {
            "model": family,
            "macro_f1": values["cv_macro_f1"],
            "recall_class_1": values["cv_recall_class_1"],
        }
        for family, values in tuned_candidates.items()
    ]
)
final_row = select_best_candidate(family_comparison)
final_family = final_row["model"]
print(
    f"Persisted three-class winner: {final_family}, "
    f"CV macro-F1={final_row['macro_f1']:.3f}, "
    f"Recall(class 1)={final_row['recall_class_1']:.3f}"
)

# %%
final_pipeline = joblib.load(PROCESSED_DIR / "a3_best_model.joblib")
validation_preds = final_pipeline.predict(X_val)
three_class_validation_metrics = evaluate_predictions(
    y_val,
    validation_preds,
)
three_class_validation_gate_passed = meets_acceptance_criteria(three_class_validation_metrics)

print("Three-class Val-2023 evidence:")
print(f"  macro-F1: {three_class_validation_metrics['macro_f1']:.3f}")
print(f"  Recall(class 1): {three_class_validation_metrics['recall_class_1']:.3f}")
print(f"  Acceptance gate passed: {three_class_validation_gate_passed}")

three_class_cm_fig = plot_confusion_matrix_heatmap(
    three_class_validation_metrics["confusion_matrix"],
    labels=["fatal", "serious", "slight"],
    title="Three-class Val-2023 confusion matrix",
)
three_class_cm_fig.show()

# %% [markdown]
# ### Persisted three-class artifacts

# %%
three_class_model_card = json.loads((PROCESSED_DIR / "a3_model_card.json").read_text())
stale_multiclass_keys = {
    "test_2024_metrics",
    "acceptance_gate_passed",
}.intersection(three_class_model_card)
if stale_multiclass_keys:
    raise ValueError(
        f"Multiclass model card contains forbidden test evidence: {sorted(stale_multiclass_keys)}"
    )
if "validation_2023_metrics" not in three_class_model_card:
    raise ValueError("Multiclass model card lacks validation-only metrics.")
print(f"Persisted model: {PROCESSED_DIR / 'a3_best_model.joblib'}")
print(f"Persisted model card: {PROCESSED_DIR / 'a3_model_card.json'}")
print("Presentation execution audits these artifacts and never rewrites them.")

# %% [markdown]
# ## 5 Empirical and arithmetic ceiling
#
# The three-class result misses the acceptance gate after the planned family,
# imbalance, and tuning search. The evidence below tests whether the gap is a
# local modelling issue or a structural limitation of the target and features.
#
# Across 19 configurations, the best observed macro-F1 is 0.424 while the gate
# requires macro-F1 of at least 0.55 and Recall(class 1) of at least 0.50. No
# tested operating point enters the feasible quadrant.

# %%
BASE = BASE_DIR
pipeline_champion = final_pipeline
y_val_proba = pipeline_champion.predict_proba(X_val)
classes = list(pipeline_champion.classes_)

print(f"Champion classes: {classes}")
print(f"Validation probability shape: {y_val_proba.shape}")
validation_argmax = evaluate_predictions(
    y_val.values,
    pipeline_champion.predict(X_val),
)
print(f"Argmax Val-2023 macro-F1: {validation_argmax['macro_f1']:.4f}")

# %%
comparison_df = pd.read_csv(BASE / "data" / "processed" / "a3_model_comparison.csv")
fig = plot_f1_recall_front(
    comparison_df,
    gate_f1=0.55,
    gate_recall=0.50,
)
fig.update_layout(title="Three-class validation front and acceptance gate")

out_path = BASE / "reports" / "figures" / "a3_f1_recall_front.html"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_path, include_plotlyjs=False)
fig.show()
print(f"Interactive front saved to {out_path}")

# %%
offsets, best_constrained_f1 = find_gate_optimal_offsets(
    y_val.values,
    y_val_proba,
    classes=classes,
    recall_gate_class=1,
    recall_gate=0.50,
)

print(f"Gate-optimal validation offsets: {offsets}")
print(f"Best validation macro-F1 under the recall gate: {best_constrained_f1:.4f}")
if offsets is not None:
    offset_class_1, offset_class_2 = offsets
    validation_logits = np.log(np.clip(y_val_proba, 1e-9, 1)).copy()
    validation_logits[:, classes.index(1)] += offset_class_1
    validation_logits[:, classes.index(2)] += offset_class_2
    y_val_pred_opt = np.array(classes)[validation_logits.argmax(axis=1)]
    val_opt = evaluate_predictions(y_val.values, y_val_pred_opt)

# %%
if offsets is not None:
    print("Gate-optimal Val-2023 conclusion:")
    print(f"  macro-F1: {val_opt['macro_f1']:.4f}")
    print(f"  Recall(class 1): {val_opt['recall_class_1']:.4f}")
    print(f"  Acceptance gate passed: {meets_acceptance_criteria(val_opt)}")
    print(
        "The gate-optimal validation operating point remains below the required macro-F1 of 0.55."
    )
else:
    print("No feasible offsets satisfy the three-class validation recall gate.")

# %% [markdown]
# ### Arithmetic ceiling
#
# With F1(class 3) near 0.72, macro-F1 of 0.55 requires classes 1 and 2 to
# average approximately 0.46 F1.
#
# Class 1 represents only 0.94% of observations. Reaching F1 of 0.46 with recall
# of at least 0.50 requires precision near 0.42, which implies roughly 90 times
# the base odds. The strongest available categorical associations remain below
# Cramér's V of 0.13. Important physical determinants such as impact speed,
# occupant age, vehicle mass, and restraint use are unavailable.
#
# The empirical front and the arithmetic requirement point to the same
# conclusion: the original gate is not reachable with this target and feature
# set.

# %% [markdown]
# ## 6 Three-class feasibility decision
#
# The three-class formulation is retained as documented background, but it does
# not continue as the primary modelling objective. Further tuning would repeat
# the same weak-signal search without changing the information available to the
# model.
#
# The analysis therefore changes the prediction target before beginning a new
# candidate search. Killed and seriously injured accidents are combined into
# KSI, while slight injuries remain the negative class. This preserves the
# safety-relevant distinction and creates a target with enough support for a
# credible validation gate.

# %% [markdown]
# ## Part II: Binary KSI model
#
# The second part begins a fresh model selection process for KSI versus slight
# injury. It does not inherit the three-class champion family.

# %% [markdown]
# ## 7 Binary target and acceptance criteria
#
# The binary target is defined as KSI for `UKATGEORIE` values 1 or 2 and slight
# injury for value 3. The acceptance gate keeps the same two objectives:
#
# - Binary macro-F1 of at least 0.55
# - Recall(KSI) of at least 0.50
#
# The binary search repeats the baseline, candidate-family, gate-aware selection,
# tuning, and threshold protocol. Balanced weighting is sufficient for the
# approximately 17-20% positive share, so the more fragile multiclass resampling
# layer is not repeated.

# %%
X_train_bin, y_train_bin = split_features_target_binary(train_df)
X_val_bin, y_val_bin = split_features_target_binary(val_df)

print(f"KSI share, Train: {y_train_bin.mean():.3f}, Val: {y_val_bin.mean():.3f}")
print(f"Rows, Train: {len(y_train_bin):,}, Val: {len(y_val_bin):,}. Test-2024 remains reserved.")

# %% [markdown]
# ## 8 Binary KSI candidate search: Stage 0 and Stage 1
#
# Random guess and majority class establish the binary floor. Logistic
# Regression provides the learned baseline.

# %%
binary_model_card = json.loads((PROCESSED_DIR / "a3_binary_model_card.json").read_text())
binary_comparison_df = pd.read_csv(PROCESSED_DIR / "a3_binary_model_comparison.csv")
candidate_registry = binary_model_card["candidate_artifacts"]
candidate_artifacts = validate_candidate_registry(
    candidate_registry,
    BASE_DIR,
)

print(f"Persisted binary configurations: {len(binary_comparison_df)}")
print(f"Validated candidate checkpoints: {len(candidate_artifacts)}")

# %% [markdown]
# ### Stage 1 tree and SVM candidates
#
# The search covers Random Forest, XGBoost, LightGBM, CatBoost, linear SVM,
# hinge-loss SGD, and an RBF SVM. Tree families and scalable linear candidates
# use the largest appropriate training scale. The RBF SVM uses an 8,000-row
# stratified sample because kernel fitting is not feasible on 1.55 million
# observations.

# %%
binary_candidate_families = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svm_linear",
    "svm_sgd",
    "svm_rbf",
]
stage1_only = binary_comparison_df[binary_comparison_df["family"].isin(binary_candidate_families)]
if len(stage1_only) != len(binary_candidate_families):
    raise ValueError("The persisted Stage 1 comparison is incomplete.")

artifact_roles = pd.Series(
    [artifact.evaluation_role for artifact in candidate_artifacts]
).value_counts()
print("Candidate artifact roles:")
print(artifact_roles.to_string())

# %% [markdown]
# ## 9 Gate-aware validation selection
#
# The same gate-aware selector is applied to the Stage 1 candidates. Baselines
# remain reference points and cannot become the champion. The candidate with the
# highest macro-F1 among those clearing Recall(KSI) of 0.50 advances to tuning.

# %%
binary_champion_row = select_best_candidate(
    stage1_only,
    recall_col="recall_ksi",
)
binary_champion_family = binary_champion_row["family"]
if binary_champion_family != binary_model_card["champion_family"]:
    raise ValueError("Persisted binary champion evidence is inconsistent.")

print(f"Binary champion family: {binary_champion_family}")
print(f"  Val-2023 macro-F1: {binary_champion_row['macro_f1']:.4f}")
print(f"  Val-2023 Recall(KSI): {binary_champion_row['recall_ksi']:.4f}")

# %%
plot_input_df = stage1_only[["model", "family", "macro_f1", "recall_ksi"]].copy()
fig = plot_binary_f1_recall_front(
    plot_input_df,
    gate_f1=0.55,
    gate_recall=0.50,
    title="Binary KSI validation front: persisted Stage 1 candidates",
)

binary_out_path = BASE_DIR / "reports" / "figures" / "a3_binary_f1_recall_front.html"
binary_out_path.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(binary_out_path, include_plotlyjs=False)
fig.show()
print(f"Interactive binary front saved to {binary_out_path}")


# %% [markdown]
# ## 10 Winning-family tuning and calibration audit
#
# This section audits the persisted single-objective and multi-objective studies.
# It reports completed and failed historical trials without scheduling new work.
# The recorded calibration refinement was attempted during the scientific run and
# was not promoted.

# %%
binary_study_name = f"binary_{binary_model_card['champion_family']}"
study_binary = optuna.load_study(
    study_name=binary_study_name,
    storage=optuna_storage,
)
binary_completed = [
    trial for trial in study_binary.trials if trial.state == optuna.trial.TrialState.COMPLETE
]
binary_failed = [
    trial for trial in study_binary.trials if trial.state == optuna.trial.TrialState.FAIL
]
if len(binary_completed) != 20:
    raise RuntimeError(f"{binary_study_name} must have 20 completed trials.")
if study_binary.best_trial.params != binary_model_card["best_hyperparameters"]:
    raise ValueError(
        "Binary model-card parameters do not match the persisted single-objective study winner."
    )

print(
    f"{binary_study_name}: {len(binary_completed)} complete, "
    f"{len(binary_failed)} failed. Persisted study audited."
)
print("No trials were scheduled.")
print(f"Recorded winning parameters: {binary_model_card['best_hyperparameters']}")

# %% [markdown]
# ### Multi-objective and calibration audit
#
# The model card records the completed refinement decision. The interrupted trial
# remains visible as failed historical evidence and is not resumed.

# %%
multiobj_study_name = f"{binary_study_name}_multiobj"
study_binary_mo = optuna.load_study(
    study_name=multiobj_study_name,
    storage=optuna_storage,
)
multiobj_completed = [
    trial for trial in study_binary_mo.trials if trial.state == optuna.trial.TrialState.COMPLETE
]
multiobj_failed = [
    trial for trial in study_binary_mo.trials if trial.state == optuna.trial.TrialState.FAIL
]
multiobj_refinement_record = binary_model_card["multiobjective_refinement"]
if not multiobj_refinement_record["attempted"]:
    raise ValueError("The persisted refinement audit is incomplete.")
if multiobj_refinement_record["promoted"]:
    raise ValueError("The model card unexpectedly promotes refinement.")

print(
    f"{multiobj_study_name}: {len(multiobj_completed)} complete, "
    f"{len(multiobj_failed)} failed. Historical state audited."
)
print("The interrupted failed trial is evidence, not pending work.")
print("Recorded decision: refinement not promoted.")
print("No trials were scheduled and no calibrated candidate was built.")

# %% [markdown]
# ## 11 Persisted champion audit, validation threshold, and one Test-2024 evaluation
#
# The fitted champion is loaded from disk. Val-2023 scores are recomputed to
# verify the recorded operating point within a narrow tolerance. The recorded
# threshold and non-promotion decision remain fixed before the notebook makes its
# sole Test-2024 prediction.

# %%
pipeline_binary_final = joblib.load(PROCESSED_DIR / "a3_binary_best_model.joblib")
best_params = binary_model_card["best_hyperparameters"]
recorded_threshold = float(binary_model_card["optimal_threshold_val_2023"])
recorded_val_metrics = binary_model_card["val_2023_metrics"]

if hasattr(pipeline_binary_final, "predict_proba"):
    y_val_scores_bin = pipeline_binary_final.predict_proba(X_val_bin)[:, 1]
else:
    y_val_scores_bin = pipeline_binary_final.decision_function(X_val_bin)

y_val_pred_bin = (y_val_scores_bin >= recorded_threshold).astype(int)
audited_val_metrics = evaluate_binary_predictions(
    y_val_bin.values,
    y_val_pred_bin,
)
for metric_name in ("macro_f1", "recall_ksi", "recall_slight"):
    if abs(audited_val_metrics[metric_name] - recorded_val_metrics[metric_name]) > 1e-9:
        raise ValueError(f"Recorded Val-2023 {metric_name} is not reproducible.")
if audited_val_metrics["confusion_matrix"] != recorded_val_metrics["confusion_matrix"]:
    raise ValueError("Recorded Val-2023 confusion matrix is not reproducible.")

best_threshold = recorded_threshold
best_f1_val = float(recorded_val_metrics["macro_f1"])
best_val_metrics = audited_val_metrics
print(f"Loaded persisted champion: {pipeline_binary_final}")
print(f"Verified Val-2023 threshold: {best_threshold:.6f}")
print(f"Verified Val-2023 macro-F1: {best_f1_val:.6f}")
print(f"Verified Val-2023 Recall(KSI): {best_val_metrics['recall_ksi']:.6f}")
print("Recorded non-promotion decision retained.")

# %%
X_test_bin, y_test_bin = split_features_target_binary(test_df)

if hasattr(pipeline_binary_final, "predict_proba"):
    y_test_scores_bin = pipeline_binary_final.predict_proba(X_test_bin)[:, 1]
else:
    y_test_scores_bin = pipeline_binary_final.decision_function(X_test_bin)

y_test_pred_bin = (y_test_scores_bin >= best_threshold).astype(int)
metrics_binary_test = evaluate_binary_predictions(
    y_test_bin.values,
    y_test_pred_bin,
)
gate_passed = meets_binary_acceptance_criteria(metrics_binary_test)
recorded_test_metrics = binary_model_card["test_2024_metrics"]
for metric_name in ("macro_f1", "recall_ksi", "recall_slight"):
    observed_delta = abs(metrics_binary_test[metric_name] - recorded_test_metrics[metric_name])
    if observed_delta > 1e-9:
        raise ValueError(f"Persisted Test-2024 {metric_name} is not reproducible.")
if metrics_binary_test["confusion_matrix"] != recorded_test_metrics["confusion_matrix"]:
    raise ValueError("Persisted Test-2024 confusion matrix is not reproducible.")

print("Binary KSI, Test-2024 metrics:")
print(f"  macro-F1: {metrics_binary_test['macro_f1']:.4f}")
print(f"  Recall(KSI): {metrics_binary_test['recall_ksi']:.4f}")
print(f"  Recall(slight): {metrics_binary_test['recall_slight']:.4f}")
print(f"  Acceptance gate passed: {gate_passed}")

confusion_fig = plot_confusion_matrix_heatmap(
    metrics_binary_test["confusion_matrix"],
    labels=["KSI", "slight"],
    title="Test-2024 confusion matrix",
)
confusion_fig.show()

# %% [markdown]
# ## 12 Binary evidence and persisted artifacts
#
# Association strength and model importance provide context for the achieved
# performance. They show how the champion combines many weak signals rather than
# relying on one dominant predictor.

# %%
from scipy.stats import chi2_contingency  # noqa: E402

from unfallatlas.features.preprocessing import ONEHOT_COLUMNS  # noqa: E402


def cramers_v(feature: pd.Series, target: pd.Series) -> float:
    """Return bias-corrected Cramér's V for two categorical series."""
    confusion = pd.crosstab(feature, target)
    n = confusion.values.sum()
    if n == 0:
        return float("nan")
    chi2 = chi2_contingency(confusion, correction=False)[0]
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2c = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - ((r - 1) ** 2) / (n - 1)
    kc = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float("nan") if denom <= 0 else float(np.sqrt(phi2c / denom))


y_binary_ksi_full = (df["UKATGEORIE"].astype(int) <= 2).astype(int)
binary_cramers_v = {
    col: cramers_v(df[col], y_binary_ksi_full) for col in ONEHOT_COLUMNS if col in df.columns
}
cramers_df = pd.DataFrame(
    [{"feature": feature, "cramers_v": value} for feature, value in binary_cramers_v.items()]
).sort_values("cramers_v")

cramers_fig = px.bar(
    cramers_df,
    x="cramers_v",
    y="feature",
    orientation="h",
    labels={"cramers_v": "Cramér's V", "feature": "Feature"},
    title="Association with the binary KSI label",
)
cramers_fig.show()
strongest = cramers_df.iloc[-1]
print(
    f"Strongest binary association: {strongest['feature']} "
    f"(Cramér's V={strongest['cramers_v']:.4f})."
)

# %%
classify_step = None
if hasattr(pipeline_binary_final, "named_steps"):
    classify_step = pipeline_binary_final.named_steps.get("classify")

binary_top_importances = None
importances = (
    getattr(classify_step, "feature_importances_", None) if classify_step is not None else None
)
if importances is not None:
    feature_names = pipeline_binary_final.named_steps["preprocess"].get_feature_names_out()
    binary_top_importances = (
        pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
    )
    importance_df = (
        binary_top_importances.rename_axis("feature")
        .rename("importance")
        .reset_index()
        .sort_values("importance")
    )
    importance_fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Model importance", "feature": "Feature"},
        title="Top feature importances: binary champion",
    )
    importance_fig.show()
    print(f"Highest model importance: {binary_top_importances.index[0]}.")
else:
    print("This pipeline does not expose tree feature importances.")

# %% [markdown]
# ### Evidence interpretation
#
# The strongest marginal association with the binary target remains modest.
# The model importance view shows that the champion distributes weight across
# road context, weather, location, and coded accident characteristics. This is
# consistent with a model extracting limited signal from several weak features.
#
# Comparable KSI studies report macro-F1 values near 0.60-0.65, although their
# inputs and evaluation protocols are not identical. The Test-2024 result should
# therefore be interpreted as credible performance for the available feature
# set, not as evidence that the missing physical determinants no longer matter.

# %% [markdown]
# ### Persisted binary model and candidate registry
#
# The champion pipeline, comparison table, and model card are persisted for
# Phase C. The model card also records every Stage 0 and Stage 1 checkpoint from
# scientific run `b1ea31e`, including its fingerprint, score interface, training
# scale, and evaluation role.

# %%
validated_candidate_artifacts = validate_candidate_registry(
    binary_model_card["candidate_artifacts"],
    BASE_DIR,
)
if binary_model_card["checkpoint_id"] != MODEL_RUN_CHECKPOINT_ID:
    raise ValueError("Model-card checkpoint does not match the audited run.")
if len(validated_candidate_artifacts) != 10:
    raise ValueError("Expected ten persisted binary candidates.")

print(f"Champion artifact: {PROCESSED_DIR / 'a3_binary_best_model.joblib'}")
print(f"Candidate registry: {len(validated_candidate_artifacts)} validated artifacts.")
print("Audit-only execution performed no artifact writes.")

# %% [markdown]
# ## 13 A³ summary and C-phase handoff
#
# The original three-class severity target did not reach its acceptance gate
# after a broad family, imbalance, tuning, and threshold search. The empirical
# front and arithmetic requirement both indicate that the missing severity
# determinants form the binding constraint.
#
# The binary KSI reformulation was evaluated through a fresh ten-candidate
# search. Random Forest won the gate-aware Val-2023 comparison, then received
# family-specific tuning and threshold selection. The persisted champion recomputation on Test-2024 reaches macro-F1 near 0.604
# and Recall(KSI) near 0.515, so both acceptance
# criteria pass.
#
# The persisted registry gives Phase C access to all ten trained candidates, not
# only the champion. Phase C can therefore compare operating behavior,
# robustness, and explanation evidence before issuing the final recommendation.
