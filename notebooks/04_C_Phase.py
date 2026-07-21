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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Unfallatlas Deutschland — C-Phase
#
# ## Position im QUA³CK-Prozess
#
# Die **C-Phase (Conclude & Compare)** schließt den QUA³CK-Zyklus für die binäre KSI-Klassifikation (getötet/schwerverletzt vs. leichtverletzt) ab. Sie trainiert **nichts neu** — sie lädt die in der A³-Phase gespeicherten Artefakte (`a3_binary_best_model.joblib`, `a3_binary_model_card.json`, `a3_binary_model_comparison.csv`) und liefert:
#
# 1. den systematischen Vergleich aller zehn Kandidaten aus dem A³-Suchlauf (ROC/PR-Kurven, Konfusionsmatrizen),
# 2. eine fehlerorientierte Diagnose (welche Slices werden systematisch verfehlt?),
# 3. die formale Go/No-Go-Prüfung gegen die Q-Phase-Gates,
# 4. eine gewichtete qualitative Bewertungsmatrix (Champion vs. die zwei nächstplatzierten Kandidaten),
# 5. eine SHAP-basierte Erklärbarkeitsanalyse (global + lokale Fallbeispiele),
# 6. den Abgleich mit der Literatur (aufbauend auf A³ §20, nicht neu hergeleitet),
# 7. eine ehrliche Limitationsdiskussion,
# 8. die finale, begründete Modellentscheidung,
# 9. die Übergabe an die K-Phase (Streamlit-App): Pipeline, Schwellenwert, Inference-Contract.
#
# **Champion (A³-Ergebnis):** `random_forest`, klassen-gewichtet, Optuna-getunt, Schwellenwert 0.4986. Test-2024: macro-F1 0.6026, Recall(KSI) 0.5255 — beide Gates (macro-F1 ≥ 0.55, Recall(KSI) ≥ 0.50) **bestanden**.

# %%
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from unfallatlas.features.preprocessing import (
    chronological_split,
    load_training_frame,
    split_features_target_binary,
)
from unfallatlas.models.evaluate import evaluate_binary_predictions
from unfallatlas.viz.metrics_viz import plot_confusion_matrix_heatmap, plot_roc_pr_curves

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
np.random.seed(42)

BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG_DIR = BASE_DIR / "reports" / "figures" / "c_phase"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = BASE_DIR / "data" / "processed"

champion_pipeline = joblib.load(PROCESSED_DIR / "a3_binary_best_model.joblib")
with open(PROCESSED_DIR / "a3_binary_model_card.json") as f:
    model_card = json.load(f)
binary_comparison_df = pd.read_csv(PROCESSED_DIR / "a3_binary_model_comparison.csv")
CHAMPION_THRESHOLD = model_card["optimal_threshold_val_2023"]

print(f"Champion family: {model_card['champion_family']}")
print(f"Threshold: {CHAMPION_THRESHOLD:.4f}")
print(f"Test-2024 macro-F1 (A³ record): {model_card['test_2024_metrics']['macro_f1']:.4f}")

# %%
df = load_training_frame(BASE_DIR)
train_df, val_df, test_df = chronological_split(df)
X_train_bin, y_train_bin = split_features_target_binary(train_df)
X_val_bin, y_val_bin = split_features_target_binary(val_df)
X_test_bin, y_test_bin = split_features_target_binary(test_df)

y_test_scores_champion = champion_pipeline.predict_proba(X_test_bin)[:, 1]
y_test_pred_champion = (y_test_scores_champion >= CHAMPION_THRESHOLD).astype(int)

sanity_metrics = evaluate_binary_predictions(y_test_bin.values, y_test_pred_champion)
recorded_metrics = model_card["test_2024_metrics"]
macro_f1_drift = abs(sanity_metrics["macro_f1"] - recorded_metrics["macro_f1"])
# A tight (1e-6) bit-exact match is not expected here: the persisted pipeline
# is unchanged, but the U-phase feature cache
# (accidents_with_weather_spatial.parquet) can have been regenerated since
# A³ was run (see the OSM tiled-fetch hotfix in the AI disclosure), which
# shifts a handful of engineered feature values slightly. A drift below 1%
# relative is treated as expected cache-refresh noise and reported
# explicitly rather than silently asserted away; anything larger would
# indicate a real problem (e.g. a stale/mismatched artifact) and must stop
# the notebook.
relative_drift = macro_f1_drift / recorded_metrics["macro_f1"]
assert relative_drift < 0.01, (
    f"Reloaded champion macro-F1 {sanity_metrics['macro_f1']:.6f} differs from the "
    f"A³-recorded {recorded_metrics['macro_f1']:.6f} by {relative_drift:.2%}, "
    "more than the 1% cache-refresh tolerance — investigate before proceeding."
)
print(
    f"Sanity check passed: reloaded champion macro-F1 {sanity_metrics['macro_f1']:.4f} vs. "
    f"A³-recorded {recorded_metrics['macro_f1']:.4f} (relative drift {relative_drift:.2%}, "
    "within the 1% cache-refresh tolerance)."
)
print(sanity_metrics)

# %% [markdown]
# ## 1 — Systematischer Modellvergleich
#
# Alle zehn Kandidaten aus dem A³-Suchlauf (drei Baselines, vier Tree-Ensemble-Familien, drei SVM-Varianten), bewertet auf Val-2023. ROC- und PR-Kurven sowie die Konfusionsmatrix werden für den Champion (`random_forest`) auf Test-2024 gezeigt.
#
# Die xgboost-/lightgbm-Pipelines wurden nicht persistiert (A³ speichert nur die finale Champion-Pipeline), daher können ihre ROC/PR-Kurven hier nicht aus gespeicherten Artefakten reproduziert werden — ihre macro-F1/Recall(KSI)-Werte aus `binary_comparison_df` bleiben aber der maßgebliche Vergleich und werden in §4 (qualitative Matrix) und §8 (finale Entscheidung) eingeordnet: beide Runner-ups erreichen höhere Recall(KSI)-Werte als der Champion.
#
# Die binäre Formulierung behandelt das Klassenungleichgewicht (~20/80) über Klassengewichtung plus schwellenwert-optimales Threshold-Moving (A³ §17) statt SMOTE/ADASYN — die multiclass-SMOTE/ADASYN-Vergleiche aus A³ §6 wurden durch die in A³ §11 bewiesene 3-Klassen-Obergrenze gegenstandslos.

# %%
display_cols = ["model", "family", "macro_f1", "recall_ksi", "recall_slight", "n_train"]
binary_comparison_df[display_cols].sort_values("macro_f1", ascending=False)

# %%
ax_roc, ax_pr = plot_roc_pr_curves(
    {"random_forest (champion)": (y_test_bin.values, y_test_scores_champion)},
    title_prefix="Test-2024 —",
)
ax_roc.figure.savefig(FIG_DIR / "roc_curve_champion.png", dpi=150, bbox_inches="tight")
ax_pr.figure.savefig(FIG_DIR / "pr_curve_champion.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
cm = confusion_matrix(y_test_bin, y_test_pred_champion, labels=[1, 0])
cm_ax = plot_confusion_matrix_heatmap(
    cm, labels=["KSI", "slight"], title="Champion — Test-2024 Confusion Matrix"
)
cm_ax.figure.savefig(FIG_DIR / "confusion_matrix_champion.png", dpi=150, bbox_inches="tight")
plt.show()
