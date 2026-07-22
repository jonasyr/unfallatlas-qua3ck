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
# # Unfallatlas Deutschland — C-Phase
#
# ## Position im QUA³CK-Prozess
#
# Die **C-Phase (Conclude & Compare)** schließt den QUA³CK-Zyklus für die binäre KSI-Klassifikation (getötet/schwerverletzt vs. leichtverletzt) ab. Sie trainiert nichts neu, sondern lädt die in der A³-Phase gespeicherten Artefakte (`a3_binary_best_model.joblib`, `a3_binary_model_card.json`, `a3_binary_model_comparison.csv`) und liefert:
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
# **Champion (A³-Ergebnis):** `random_forest`, klassen-gewichtet, Optuna-getunt, Schwellenwert 0.4986. Test-2024: macro-F1 0.6026, Recall(KSI) 0.5255; beide Gates (macro-F1 ≥ 0.55, Recall(KSI) ≥ 0.50) **bestanden**.

# %%
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.metrics import confusion_matrix

from unfallatlas.features.preprocessing import (
    chronological_split,
    load_training_frame,
    split_features_target_binary,
)
from unfallatlas.models.c_phase import (
    build_inference_contract,
    build_qualitative_matrix,
    compute_error_slices,
    decode_slice_label,
    humanize_feature_name,
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
    "more than the 1% cache-refresh tolerance: investigate before proceeding."
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
# Die xgboost-/lightgbm-Pipelines wurden nicht persistiert (A³ speichert nur die finale Champion-Pipeline), daher können ihre ROC/PR-Kurven hier nicht aus gespeicherten Artefakten reproduziert werden; ihre macro-F1/Recall(KSI)-Werte aus `binary_comparison_df` bleiben aber der maßgebliche Vergleich und werden in §4 (qualitative Matrix) und §8 (finale Entscheidung) eingeordnet: beide Runner-ups erreichen höhere Recall(KSI)-Werte als der Champion.
#
# Die binäre Formulierung behandelt das Klassenungleichgewicht (~20/80) über Klassengewichtung plus schwellenwert-optimales Threshold-Moving (A³ §17) statt SMOTE/ADASYN; die multiclass-SMOTE/ADASYN-Vergleiche aus A³ §6 wurden durch die in A³ §11 bewiesene 3-Klassen-Obergrenze gegenstandslos.

# %%
display_cols = ["model", "family", "macro_f1", "recall_ksi", "recall_slight", "n_train"]
binary_comparison_df[display_cols].sort_values("macro_f1", ascending=False)

# %%
roc_fig, pr_fig = plot_roc_pr_curves(
    {"random_forest (champion)": (y_test_bin.values, y_test_scores_champion)},
    title_prefix="Test-2024",
)
roc_fig.write_html(FIG_DIR / "roc_curve_champion.html", include_plotlyjs=True)
pr_fig.write_html(FIG_DIR / "pr_curve_champion.html", include_plotlyjs=True)
roc_fig.show()
pr_fig.show()

# %%
cm = confusion_matrix(y_test_bin, y_test_pred_champion, labels=[1, 0])
cm_fig = plot_confusion_matrix_heatmap(
    cm, labels=["KSI", "slight"], title="Champion: Test-2024 Confusion Matrix"
)
cm_fig.write_html(FIG_DIR / "confusion_matrix_champion.html", include_plotlyjs=True)
cm_fig.show()

# %% [markdown]
# **Interpretation:** Die ROC-Kurve liegt deutlich über der Zufalls-Diagonale, was ein
# brauchbares, aber nicht sehr starkes Trennvermögen anzeigt (konsistent mit der schwachen
# Merkmalsassoziation aus §6). Die Precision-Recall-Kurve fällt bei steigendem Recall spürbar ab,
# was bei diesem starken Klassenungleichgewicht (~20 % KSI) typisch ist und den in §0 berichteten
# Recall(KSI)-Wert (≈0,52) einordnet: Über die Hälfte der tatsächlichen KSI-Fälle wird erkannt, aber
# nicht durchgängig. Die Konfusionsmatrix macht das konkret: Der Champion erkennt gut die Hälfte
# der echten KSI-Fälle korrekt (oben links), übersieht aber die andere Hälfte (oben rechts, False
# Negatives); umgekehrt werden auch etliche leichte Unfälle fälschlich als KSI eingestuft (unten
# links, False Positives). Beide Fehlerarten werden in §2 nach Slice aufgeschlüsselt.
#
# ## 2 — Fehleranalyse nach Slices
#
# False Negatives (übersehene KSI-Fälle) und False Positives, aufgeschlüsselt nach Unfalltyp (`UART`), dominanter OSM-Straßenklasse, Straßenzustand (`STRZUSTAND`), Lichtverhältnissen (`ULICHTVERH`), Niederschlagskategorie (`_precip_bucket`, aus `dwd_precip_mm` abgeleitet) und Tageszeit (`USTUNDE`, Unfallstunde 0–23), um zu prüfen, ob Fehler systematisch in bestimmten Teilgruppen auftreten oder gleichmäßig verteilt sind.

# %%
slice_columns = [
    "UART",
    "osm_dominant_road_class",
    "STRZUSTAND",
    "ULICHTVERH",
    "_precip_bucket",
    "USTUNDE",
]
slice_frame = test_df[slice_columns].reset_index(drop=True)

error_slice_df = compute_error_slices(
    pd.Series(y_test_bin.values), pd.Series(y_test_pred_champion), slice_frame, slice_columns
)
# Decoded "Spalte: Wertlabel" per row (e.g. "Unfallart: Zzs. vorausfahrendes
# Fz. (Auffahrunfall)") so every table/chart below shows the human-readable
# category directly, never a bare code the reader has to look up elsewhere.
error_slice_df["label"] = [
    decode_slice_label(col, value)
    for col, value in zip(error_slice_df["slice_column"], error_slice_df["slice_value"])
]
error_display_cols = [
    "label",
    "n",
    "n_false_negative",
    "n_false_positive",
    "false_negative_rate",
    "false_positive_rate",
]
error_slice_df[error_display_cols].sort_values("false_negative_rate", ascending=False).head(20)

# %%
plot_df = error_slice_df[error_slice_df["n"] >= 100].nlargest(15, "false_negative_rate")

error_slice_fig = go.Figure(
    go.Bar(
        x=plot_df["false_negative_rate"],
        y=plot_df["label"],
        orientation="h",
        marker_color="#528AAE",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    )
)
error_slice_fig.update_layout(
    title="Höchste False-Negative-Raten nach Slice (n ≥ 100)",
    xaxis_title="False-Negative-Rate",
    xaxis_tickformat=".0%",
    template="plotly_white",
    yaxis=dict(autorange="reversed"),
    height=520,
)
error_slice_fig.write_html(FIG_DIR / "error_slices_fn_rate.html", include_plotlyjs=True)
error_slice_fig.show()

# %% [markdown]
# **Beobachtung:** Die höchsten False-Negative-Raten treten bei `UART`-Kategorien auf: `UART=1` (Kollision mit haltendem/parkendem Fahrzeug, 89,0 % FN-Rate, n=14.673), `UART=3` (Seitenkollision, 74,1 %) und `UART=2` (Auffahrunfall, 71,5 %). Auch `STRZUSTAND=2` (Winterglätte, 68,5 %, wenn auch mit kleinerem n=4.731) und `UART=5` (Abbiege-/Einbiegeunfall, der häufigste Unfalltyp, 68,0 %) liegen weit oben. Bemerkenswert: Gerade die von A³ §20 als stärkstes Einzelmerkmal identifizierte Variable `UART` (Cramér's V=0,1801) dominiert auch hier die Fehlerliste: selbst das informativste verfügbare Merkmal reicht nicht aus, um KSI-Fälle innerhalb dieser Unfalltypen zuverlässig zu erkennen. Das deckt sich mit der in §6 aufgegriffenen Feature-Obergrenze: Die Fehler sind nicht zufällig verteilt, sondern konzentrieren sich systematisch dort, wo die verfügbaren Merkmale am wenigsten trennscharf sind.
#
# Die beiden neuen Dimensionen zeigen ein deutlich schwächeres, aber nicht triviales Muster: Nach Tageszeit (`USTUNDE`) liegt die FN-Rate im morgendlichen Berufsverkehr (7–9 Uhr) mit 56,7–60,7 % spürbar über dem Durchschnitt und fällt danach kontinuierlich ab (10 Uhr: 50,8 %); ein Hinweis, dass der Champion KSI-Fälle im dichten Frühverkehr etwas häufiger übersieht als in verkehrsärmeren Stunden. Nach Niederschlagskategorie (`_precip_bucket`) zeigt sich dagegen praktisch kein Unterschied (dry: 47,2 % vs. light 0–5 mm: 48,7 % FN-Rate); in Test-2024 kommen ohnehin nur diese zwei Kategorien vor (moderate/heavy: n=0), sodass Niederschlag hier keine systematische Fehlerquelle ist, anders als Unfalltyp, Straßenzustand oder Tageszeit.

# %% [markdown]
# ## 3 — Formale KPI-Validierung: Go/No-Go
#
# Explizite Prüfung des Champions gegen die in der Q-Phase festgelegten Akzeptanzkriterien für die binäre KSI-Formulierung, auf Basis der in §0 frisch berechneten Test-2024-Metriken (`sanity_metrics`).

# %%
champion_val_row = next(
    r for r in model_card["stage0_1_comparison"] if r["family"] == model_card["champion_family"]
)

gate_table = pd.DataFrame(
    [
        {
            "Gate": "macro-F1 >= 0.55",
            "Val-2023": model_card["val_2023_macro_f1"],
            "Test-2024": sanity_metrics["macro_f1"],
            "Passed": sanity_metrics["macro_f1"] >= 0.55,
        },
        {
            "Gate": "Recall(KSI) >= 0.50",
            "Val-2023": champion_val_row["recall_ksi"],
            "Test-2024": sanity_metrics["recall_ksi"],
            "Passed": sanity_metrics["recall_ksi"] >= 0.50,
        },
    ]
)
gate_overall_pass = bool(gate_table["Passed"].all())
print(f"Overall gate PASSED: {gate_overall_pass}")
gate_table

# %% [markdown]
# ## 4 — Qualitative Bewertungsmatrix
#
# Reine Metriken (macro-F1, Recall(KSI)) reichen nicht aus, um zwischen dem Champion und den zwei nächstplatzierten Kandidaten zu entscheiden: die Runner-ups (`xgboost`, `lightgbm`) haben höhere Recall(KSI)-Werte. Diese gewichtete Matrix berücksichtigt zusätzlich Inferenzgeschwindigkeit, Interpretierbarkeit, Robustheit gegenüber fehlenden OSM/DWD-Features und Trainingskosten.
#
# **Gewichtung:** macro-F1 und Recall(KSI) je 30 % (Kernmetriken der Q-Phase-Gates), die übrigen vier Kriterien je 10 %.

# %%
_latency_sample = X_test_bin.sample(n=1000, random_state=42)
_start = time.perf_counter()
champion_pipeline.predict_proba(_latency_sample)
_champion_latency_ms_per_1k = (time.perf_counter() - _start) * 1000
print(f"Champion latency: {_champion_latency_ms_per_1k:.1f} ms per 1,000 rows")

# %%
# Robustness check: all three families share the exact same ColumnTransformer
# preprocessing (build_preprocessor() is passed in externally to every
# build_*_pipeline() in src/unfallatlas/models/*.py), so their handling of
# missing OSM/DWD input is identical, not a per-family trait. Verify this
# empirically instead of asserting it: null every OSM/DWD column on a small
# batch and confirm the champion pipeline still predicts without raising
# (numeric OSM/DWD columns are median-imputed, the one OSM categorical
# column is constant-imputed to "unknown" - see build_preprocessor()).
_osm_dwd_cols = [
    c for c in X_test_bin.columns if c.startswith(("osm_", "dwd_")) or c == "_precip_bucket"
]
_robustness_probe = X_test_bin.iloc[:50].copy()
for _col in _osm_dwd_cols:
    _robustness_probe[_col] = np.nan if _robustness_probe[_col].dtype.kind in "fc" else None
try:
    champion_pipeline.predict_proba(_robustness_probe)
    ROBUSTNESS_OK = True
except Exception as exc:
    ROBUSTNESS_OK = False
    print(f"Robustness probe FAILED: {exc}")
# Identical across families since it is a preprocessing-pipeline property,
# not a classifier property.
ROBUSTNESS_SCORE = 0.8 if ROBUSTNESS_OK else 0.2
print(
    f"Pipeline tolerates fully-missing OSM/DWD input ({len(_osm_dwd_cols)} columns nulled): "
    f"{ROBUSTNESS_OK} -> robustness_score={ROBUSTNESS_SCORE} for all three families"
)

# %%
champion_row = binary_comparison_df[
    binary_comparison_df["family"] == model_card["champion_family"]
].iloc[0]
xgboost_row = binary_comparison_df[binary_comparison_df["family"] == "xgboost"].iloc[0]
lightgbm_row = binary_comparison_df[binary_comparison_df["family"] == "lightgbm"].iloc[0]

# Interpretability: random_forest exposes native feature_importances_ and is a
# bagged-tree ensemble (each tree independently traceable); xgboost/lightgbm
# are boosted ensembles (feature_importances_ also available, but individual
# trees correct previous residuals rather than voting independently, making
# per-prediction path tracing less direct without SHAP). Scored 0-1, champion
# favoured for its direct TreeExplainer compatibility used in §5.
# Robustness: ROBUSTNESS_SCORE (computed above) is identical for all three -
# it reflects the shared ColumnTransformer preprocessing, not a per-family
# difference.
# Training cost: Optuna trial count from the shared A³ search budget
# (provenance.optuna_trials applies to the whole binary Stage-1 search, so
# it is identical across families here: a genuine shared-cost fact, not an
# invented per-family estimate).
qualitative_rows = [
    {
        "model": "random_forest (champion)",
        "macro_f1": champion_row["macro_f1"],
        "recall_ksi": champion_row["recall_ksi"],
        "latency_ms_per_1k": _champion_latency_ms_per_1k,
        "interpretability_score": 0.8,
        "robustness_score": ROBUSTNESS_SCORE,
        "training_cost_score": model_card["provenance"]["optuna_trials"],
    },
    {
        "model": "xgboost",
        "macro_f1": xgboost_row["macro_f1"],
        "recall_ksi": xgboost_row["recall_ksi"],
        "latency_ms_per_1k": _champion_latency_ms_per_1k,
        "interpretability_score": 0.6,
        "robustness_score": ROBUSTNESS_SCORE,
        "training_cost_score": model_card["provenance"]["optuna_trials"],
    },
    {
        "model": "lightgbm",
        "macro_f1": lightgbm_row["macro_f1"],
        "recall_ksi": lightgbm_row["recall_ksi"],
        "latency_ms_per_1k": _champion_latency_ms_per_1k,
        "interpretability_score": 0.6,
        "robustness_score": ROBUSTNESS_SCORE,
        "training_cost_score": model_card["provenance"]["optuna_trials"],
    },
]
qualitative_matrix_df = build_qualitative_matrix(qualitative_rows)
qualitative_matrix_df

# %% [markdown]
# **Hinweis zur Latenz, Robustheit und den Trainingskosten:** Nur die Champion-Pipeline ist als Artefakt gespeichert (`a3_binary_best_model.joblib`); xgboost/lightgbm wurden nicht auf dem vollen Trainingsset refittet und persistiert, daher kann ihre Inferenzlatenz hier nicht separat gemessen werden; der Platzhalterwert (identisch zum Champion) wird explizit als Limitation benannt statt stillschweigend als exakter Wert behandelt. Der Robustheits-Score ist für alle drei Familien identisch (`ROBUSTNESS_SCORE`, oben empirisch geprüft): Alle drei teilen sich dieselbe `ColumnTransformer`-Preprocessing-Pipeline, daher ist die Behandlung fehlender OSM/DWD-Werte eine Eigenschaft der Pipeline, nicht des Klassifikators. Der Trainingskosten-Score (`optuna_trials`) bezieht sich auf das gemeinsame Suchbudget des gesamten binären Stage-1-Laufs und ist daher ebenfalls für alle drei Familien identisch; auch das ist ein echter, dokumentierter Fakt aus der Provenienz und keine erfundene Pro-Familie-Schätzung. Alle drei Kriterien tragen wegen fehlender Varianz nicht zur Rangfolge bei; die Entscheidung stützt sich damit primär auf macro-F1, Recall(KSI) und Interpretierbarkeit.

# %% [markdown]
# ## 5 — SHAP-Erklärbarkeit
#
# `TreeExplainer` auf einer stratifizierten Stichprobe von 5.000 Zeilen aus Test-2024 (2.500 pro Klasse); die vollen ~223.000 Test-2024-Zeilen sind für SHAP nicht praktikabel. Zusätzlich wird `approximate=True` (Saabas-Algorithmus) verwendet: Der Champion hat 180 Bäume mit exakt Tiefe 23 und im Schnitt ~41.355 Blättern pro Baum (insgesamt ~7,4 Mio. Blätter); für Bäume dieser Größe ist die exakte SHAP-Berechnung (Komplexität ~O(Blätter·Tiefe²)) empirisch bestätigt unpraktikabel (ein Testlauf mit 50 Zeilen ohne `approximate=True` lief über 10 Minuten, ohne zu terminieren; mit `approximate=True` dauerten 500 Zeilen 0,11 Sekunden). Diese Stichprobengröße und die Approximation sind bewusste, hier dokumentierte Entscheidungen, keine stillschweigenden Kürzungen. Zunächst die globale Sicht (Summary/Beeswarm + mittlere absolute SHAP-Werte), danach vier konkrete Fallbeispiele.

# %%
sample_idx = (
    pd.Series(y_test_bin.values)
    .groupby(y_test_bin.values)
    .sample(n=2500, random_state=42)  # 2,500 per class = 5,000 total, stratified
    .index
)
shap_sample_X_raw = X_test_bin.iloc[sample_idx].reset_index(drop=True)
shap_sample_y = y_test_bin.iloc[sample_idx].reset_index(drop=True)

preprocessor = champion_pipeline[:-1]
classifier = champion_pipeline[-1]
shap_sample_X = pd.DataFrame(
    preprocessor.transform(shap_sample_X_raw),
    columns=preprocessor.get_feature_names_out(),
)
# ColumnTransformer stacks heterogeneous encoders (OneHotEncoder returns
# bool, passthrough returns int8, target/cyclic encoders return float) into
# a single object-dtype array. That silently breaks SHAP's colour-by-
# feature-value coding in the beeswarm plot below (all points render gray
# instead of the usual low->high gradient) even though every value is
# numeric. Cast to float64 explicitly - verified safe (0 NaNs introduced).
shap_sample_X = shap_sample_X.astype(float)

explainer = shap.TreeExplainer(classifier)
# approximate=True (Saabas algorithm): the exact TreeExplainer algorithm is
# impractical for this champion's tree size (180 trees, depth 23, ~7.4M
# leaves total); see the markdown above for the empirical timing that
# motivated this choice. check_additivity=False because the approximate
# algorithm does not guarantee exact additivity to the model's raw output.
shap_values = explainer.shap_values(shap_sample_X, approximate=True, check_additivity=False)
# For binary sklearn classifiers, shap_values may be a list [class0, class1]
# or a single 2D array depending on the shap version pinned; handle both.
shap_values_ksi = shap_values[1] if isinstance(shap_values, list) else shap_values
if shap_values_ksi.ndim == 3:
    # shap>=0.45 TreeExplainer on binary classifiers can return shape
    # (n_samples, n_features, n_classes) instead of a list; select class 1.
    shap_values_ksi = shap_values_ksi[:, :, 1]
print(f"shap_values_ksi shape: {shap_values_ksi.shape}, shap_sample_X shape: {shap_sample_X.shape}")

# %%
shap_importance = pd.Series(
    np.abs(shap_values_ksi).mean(axis=0), index=shap_sample_X.columns
).sort_values(ascending=False)

N_BEESWARM_FEATURES = 20
top_features = shap_importance.head(N_BEESWARM_FEATURES).index[::-1]  # most important on top
feature_col_idx = {col: i for i, col in enumerate(shap_sample_X.columns)}
_jitter_rng = np.random.default_rng(42)

beeswarm_fig = go.Figure()
for row, feat in enumerate(top_features):
    feat_vals = shap_sample_X[feat].to_numpy()
    shap_vals = shap_values_ksi[:, feature_col_idx[feat]]
    val_min, val_max = feat_vals.min(), feat_vals.max()
    span = (val_max - val_min) or 1.0
    color_vals = (feat_vals - val_min) / span
    jitter = _jitter_rng.uniform(-0.35, 0.35, size=len(shap_vals))
    beeswarm_fig.add_trace(
        go.Scatter(
            x=shap_vals,
            y=row + jitter,
            mode="markers",
            marker=dict(
                color=color_vals,
                colorscale="RdBu_r",
                size=4,
                opacity=0.65,
                showscale=(row == 0),
                colorbar=dict(
                    title="Feature-<br>wert", tickvals=[0, 1], ticktext=["niedrig", "hoch"], x=1.02
                )
                if row == 0
                else None,
            ),
            name=humanize_feature_name(feat),
            hovertemplate=f"{humanize_feature_name(feat)}<br>SHAP=%{{x:.3f}}<extra></extra>",
            showlegend=False,
        )
    )
beeswarm_fig.update_layout(
    title=f"SHAP Summary (Beeswarm), Top {N_BEESWARM_FEATURES} Features",
    xaxis_title="SHAP-Wert (Einfluss auf KSI-Score)",
    yaxis=dict(
        tickmode="array",
        tickvals=list(range(N_BEESWARM_FEATURES)),
        ticktext=[humanize_feature_name(f) for f in top_features],
    ),
    template="plotly_white",
    height=700,
    margin=dict(r=120),
)
beeswarm_fig.write_html(FIG_DIR / "shap_summary_beeswarm.html", include_plotlyjs=True)
beeswarm_fig.show()

# %%
N_BAR_FEATURES = 15
bar_data = shap_importance.head(N_BAR_FEATURES).iloc[::-1]

bar_fig = go.Figure(
    go.Bar(
        x=bar_data.values,
        y=[humanize_feature_name(f) for f in bar_data.index],
        orientation="h",
        marker_color="#528AAE",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    )
)
bar_fig.update_layout(
    title=f"Mittlere absolute SHAP-Werte (Top {N_BAR_FEATURES})",
    xaxis_title="mean(|SHAP value|)",
    template="plotly_white",
    height=500,
)
bar_fig.write_html(FIG_DIR / "shap_importance_bar.html", include_plotlyjs=True)
bar_fig.show()

# %% [markdown]
# **Interpretation:** Im Beeswarm-Plot verschiebt Rot (hoher Feature-Wert) den SHAP-Wert für die
# meisten Top-Features klar in eine Richtung, Blau (niedriger Wert) in die andere; eine breite,
# durchmischte Punktwolke statt einer klaren Trennung würde auf ein schwaches Merkmal hindeuten.
# Der Balkenplot bestätigt die Rangfolge über den Mittelwert der absoluten SHAP-Beiträge: Kein
# einzelnes Feature dominiert (§6 ordnet diese Rangfolge gegen die Literatur ein). Die vier
# folgenden Fallbeispiele zeigen, wie sich dieses globale Muster auf einzelne Vorhersagen auswirkt.

# %%
shap_sample_pred_proba = champion_pipeline.predict_proba(shap_sample_X_raw)[:, 1]
shap_sample_pred = (shap_sample_pred_proba >= CHAMPION_THRESHOLD).astype(int)

is_tp = (shap_sample_y.values == 1) & (shap_sample_pred == 1)
is_fn = (shap_sample_y.values == 1) & (shap_sample_pred == 0)
is_fp = (shap_sample_y.values == 0) & (shap_sample_pred == 1)
is_tn = (shap_sample_y.values == 0) & (shap_sample_pred == 0)

case_indices = {}
for name, mask in [
    ("true_positive_ksi", is_tp),
    ("false_negative_ksi", is_fn),
    ("false_positive_slight", is_fp),
    ("true_negative", is_tn),
]:
    matches = np.where(mask)[0]
    if len(matches) == 0:
        print(f"WARNING: no examples found for {name} in this sample, skipping")
        continue
    case_indices[name] = matches[0]
print(case_indices)

# %%
expected_value = explainer.expected_value
expected_value_ksi = (
    expected_value[1]
    if isinstance(expected_value, (list, np.ndarray)) and len(np.atleast_1d(expected_value)) > 1
    else expected_value
)

N_WATERFALL_FEATURES = 10
for name, idx in case_indices.items():
    contributions = pd.Series(shap_values_ksi[idx], index=shap_sample_X.columns)
    top_contrib = contributions.reindex(
        contributions.abs().sort_values(ascending=False).index
    ).head(N_WATERFALL_FEATURES)
    other_sum = contributions.drop(top_contrib.index).sum()

    labels = [
        "Basiswert",
        *(humanize_feature_name(f) for f in top_contrib.index),
        "Übrige Features",
        "Vorhersage",
    ]
    values = [expected_value_ksi, *top_contrib.tolist(), other_sum, 0]
    measures = ["absolute", *(["relative"] * (len(top_contrib) + 1)), "total"]

    waterfall_fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgba(120, 120, 120, 0.4)"}},
            increasing={"marker": {"color": "#E06C75"}},
            decreasing={"marker": {"color": "#56B6C2"}},
            totals={"marker": {"color": "#528AAE"}},
        )
    )
    waterfall_fig.update_layout(
        title=f"SHAP Waterfall: {name}",
        yaxis_title="Score-Beitrag (Richtung KSI)",
        template="plotly_white",
        height=480,
        xaxis_tickangle=-35,
    )
    waterfall_fig.write_html(FIG_DIR / f"shap_waterfall_{name}.html", include_plotlyjs=True)
    waterfall_fig.show()

# %% [markdown]
# **Fallbeispiele:** Bei der **True-Positive-KSI** (korrekt erkannter KSI-Fall) treibt `IstKrad` (Motorradbeteiligung) den Score am stärksten Richtung KSI (SHAP=+0,13); mit Abstand der größte Einzelbeitrag unter allen vier Fällen, konsistent mit dem bekannten hohen Verletzungsrisiko für Motorradfahrer. Bei der **False-Negative-KSI** (übersehener KSI-Fall) sind die Beiträge insgesamt deutlich schwächer (größter Betrag nur 0,036 für `UKREIS_target_enc`, und dieser zeigt sogar in die falsche Richtung); genau das Muster, das die niedrige Recall(KSI) erklärt: Es fehlt nicht an einem falsch gewichteten Feature, sondern schlicht an einem hinreichend starken Signal in den verfügbaren Merkmalen für diesen Fall. Beim **False-Positive-Slight** (fälschlich als KSI eingestuft) ziehen `UTYP1_1` (Fahrunfall), `osm_road_density` und `osm_way_count` gemeinsam Richtung KSI (SHAP zwischen +0,05 und +0,06 je Feature); ein Muster aus mehreren mittelstarken OSM-/Unfalltyp-Signalen, das in diesem Fall in die falsche Richtung zeigt. Bei der **True Negative** dominiert `UART_2` (Auffahrunfall) mit SHAP=-0,14 klar Richtung "leicht"; der stärkste Einzelbeitrag unter den korrekten Klassifikationen. Alle vier Fälle stützen sich auf dieselbe Merkmalsfamilie wie die globale Rangfolge oben (`osm_way_count`, `IstKrad`, `UTYP1_1`, `osm_road_density`, `UART_2` sind die fünf global wichtigsten Features); es gibt kein verborgenes, in den Einzelfällen dominierendes Merkmal, das in der globalen Sicht fehlen würde.

# %% [markdown]
# ## 6 — Abgleich mit der Literatur
#
# A³ §20 hat Cramér's V direkt gegen das **binäre** KSI-Label neu berechnet (nicht gegen das ursprüngliche 3-Klassen-`UKATGEORIE`, für das die U-Phase, §6, bereits eine Assoziationsobergrenze von ≤0,13 dokumentiert hatte): `UART` (Unfallart) ist mit **0,1801** das stärkste Einzelmerkmal für die binäre Klassifikation, gefolgt von `UTYP1` mit 0,1505; `ULICHTVERH` und `STRZUSTAND` liegen beide unter 0,03. Selbst das stärkste binäre Merkmal bleibt damit deutlich unter dem für starkes Klassifikationssignal üblichen Bereich von ~0,3–0,5; die binäre Reformulierung erhöht zwar die erreichbare Vorhersagegüte gegenüber der 3-Klassen-Formulierung (A³ §11), löst aber nicht das zugrundeliegende Problem schwacher Merkmalsassoziation.
#
# Diese SHAP-Analyse ergänzt eine dritte, unabhängige Sicht auf dieselbe Frage: die mittleren absoluten SHAP-Werte über die 5.000er-Stichprobe:

# %%
shap_importance.rename(humanize_feature_name).head(15)

# %% [markdown]
# Die global wichtigsten SHAP-Features sind `osm_way_count`, `IstKrad`, `UTYP1_1`, `osm_road_density`, `UART_2`, `IstPKW`, `UKREIS_target_enc`, `osm_maxspeed_mean`; eine Mischung aus OSM-Straßenkontext, Fahrzeugtyp-Flags und Unfalltyp-Kategorien, nicht eine einzelne dominante Variable. Das deckt sich mit A³ §20/§21: Der Champion stützt sich stärker auf OSM-Straßenkontext- und Geo-Features als primär auf die assoziationsstärksten `UART`/`UTYP1`-Codes; ein Modell, das schwaches Signal über viele Merkmale hinweg extrahiert, statt sich auf ein dominantes Prädiktor zu verlassen.
#
# Das erreichte Test-2024 macro-F1 (0,6039 in dieser Neuberechnung; 0,6026 im A³-Rekord) liegt im von der Q-Phase zitierten Literaturbereich für vergleichbare KSI-vs.-leicht-Klassifikation (Santos 2022 ≈ 0,60, Pakgohar 2021 ≈ 0,62, Schlößler 2024 ≈ 0,65); konsistent mit, nicht unterhalb des Stands der Technik auf diesem Feature-Set.

# %% [markdown]
# ## 7 — Limitationen
#
# - **Selektionsbias:** Der Unfallatlas erfasst nur polizeilich gemeldete Unfälle; leichte Unfälle ohne Polizeibeteiligung fehlen systematisch, was die tatsächliche Grundgesamtheit verzerrt.
# - **Fehlende physische Determinanten:** Aufprallgeschwindigkeit, Gurtnutzung, Insassenalter und Fahrzeugmasse (die stärksten bekannten Prädiktoren für Verletzungsschwere in der Literatur) liegen nicht im öffentlichen Unfallatlas vor, sondern in zugriffsbeschränkten Destatis-Personen-/Fahrzeugmikrodaten (siehe A³ §11/§19 `gate_reformulation_reason`).
# - **Korrelation ≠ Kausalität:** SHAP-Werte und Feature-Importances zeigen Assoziationen, keine kausalen Effekte; z. B. sagt eine hohe SHAP-Bedeutung von OSM-Straßenkontext-Features nichts darüber aus, ob bauliche Eingriffe die KSI-Rate kausal senken würden.
# - **OSM-Features sind zeitlich nicht versioniert:** Die OSM-Straßenkontext-Features spiegeln das heutige Straßennetz wider und werden einheitlich auf alle Unfalljahre (2016–2024) angewendet; eine dokumentierte, akzeptierte Näherung (U-Phase §8.8, siehe Glossar), keine Leckage-Quelle, aber eine Einschränkung der historischen Genauigkeit.
# - **Geografische/zeitliche Abdeckung:** Trainingsdaten 2016–2022, Validierung 2023, Test 2024; Verallgemeinerung auf zukünftige Jahre oder auf Regionen mit strukturell anderer Infrastruktur ist nicht geprüft.
# - **Schwellenwert-Sensitivität:** Der gate-optimale Schwellenwert (0,4986) wurde auf Val-2023 gewählt; siehe §3 für die Gate-Ergebnisse bei diesem Schwellenwert; eine Verschiebung würde den Recall(KSI)/macro-F1-Tradeoff entlang der in §1 gezeigten Kurven verändern.
# - **Restliches Klassenungleichgewicht:** Trotz Klassengewichtung und Threshold-Moving verfehlt der Champion Recall(KSI) gegenüber den Runner-ups (§1/§4); ein bewusster Tradeoff zugunsten von macro-F1 (§4/§8), nicht ein ungelöstes technisches Problem.

# %% [markdown]
# ## 8 — Finale Modellentscheidung
#
# **Synthese:** Der formale Gate-Check (§3) ist für beide Kriterien **bestanden** (macro-F1 0,6039 ≥ 0,55; Recall(KSI) 0,5151 ≥ 0,50). Die qualitative Bewertungsmatrix (§4) bestätigt `random_forest` als Champion mit dem höchsten gewichteten Score (0,600 vs. 0,518 für xgboost und 0,500 für lightgbm), trotz niedrigerer Recall(KSI) als beide Runner-ups (§1); der Tradeoff zugunsten von macro-F1 ist durch die 30 %/30 %-Gewichtung explizit gemacht, nicht implizit angenommen. Latenz, Robustheit gegenüber fehlenden OSM/DWD-Features und Trainingskosten sind für alle drei Familien identisch (§4) und tragen daher nicht zur Rangfolge bei; die Entscheidung stützt sich primär auf macro-F1, Recall(KSI) und Interpretierbarkeit. SHAP (§5) und der Literaturabgleich (§6) zeigen ein Modell, das auf breit verteilten, schwach assoziierten Features (OSM-Straßenkontext, Fahrzeugtyp-Flags, Unfalltyp) basiert statt auf einem einzelnen dominanten Prädiktor; konsistent mit der in A³ §11/§20 belegten Feature-Obergrenze (stärkste binäre Assoziation `UART`=0,1801, weit unter dem Bereich starken Klassifikationssignals). Die Fehleranalyse (§2) zeigt, dass die verbleibenden Fehler systematisch dort auftreten, wo die verfügbaren Merkmale am wenigsten trennscharf sind (Unfalltyp, Straßenzustand, morgendlicher Berufsverkehr); kein Hinweis auf eine behebbare, aber übersehene Schwäche des Champions.
#
# **Entscheidung:** `random_forest` (Schwellenwert 0,4986) bleibt der bestätigte Champion für die K-Phase.

# %% [markdown]
# ## 9 — Übergabe an die K-Phase
#
# Vollständiges Artefaktpaket für die Streamlit-App: die bereits gespeicherte Pipeline (`a3_binary_best_model.joblib`), der Schwellenwert, und ein neuer Inference-Contract, der alle erforderlichen Eingabespalten mit Datentyp, gültigem Wertebereich bzw. Kategorienliste (aus den echten Trainingsdaten abgeleitet, nicht angenommen) und Herkunft (Unfallatlas roh/abgeleitet, DWD-Wetteranreicherung oder OSM-Straßenkontext-Anreicherung, jeweils aus der U-Phase) auflistet, damit die K-Phase-Implementierung nichts aus den Notebooks neu ableiten muss.

# %%
feature_columns = X_train_bin.columns.tolist()
dtypes = {col: str(dtype) for col, dtype in X_train_bin.dtypes.items()}

inference_contract = build_inference_contract(feature_columns, dtypes, model_card, X_train_bin)
with open(PROCESSED_DIR / "c_phase_inference_contract.json", "w") as f:
    json.dump(inference_contract, f, indent=2)

print(f"Inference contract written: {PROCESSED_DIR / 'c_phase_inference_contract.json'}")
print(f"Required columns: {len(inference_contract['required_columns'])}")
print(f"Model artifact: {inference_contract['model_path']} (unchanged, re-confirmed present)")
assert (PROCESSED_DIR / "a3_binary_best_model.joblib").exists()

# %% [markdown]
# ## Zusammenfassung der C-Phase
#
# **Vom 3-Klassen- zum binären Ziel:** Das Projekt startete mit `UKATGEORIE` (3 Klassen: getötet/schwerverletzt/leichtverletzt) als Zielvariable. Die U-Phase (§6, Cramér's-V-Matrix) fand für die stärksten Merkmale gegen `UKATGEORIE` eine Assoziation von höchstens 0,13; A³ §11 baute darauf auf und bewies empirisch die daraus resultierende Klassifikations-Obergrenze für das 3-Klassen-Problem: kein verfügbares Merkmal im öffentlichen Unfallatlas trennt die drei Klassen ausreichend, weil die stärksten bekannten Prädiktoren (Aufprallgeschwindigkeit, Gurtnutzung u. Ä., §7) nicht öffentlich vorliegen. Die binäre Reformulierung (KSI = getötet/schwerverletzt vs. leichtverletzt) wurde daraufhin adoptiert: Sie erhöht die stärkste Einzelmerkmal-Assoziation auf 0,1801 (`UART`, A³ §20, §6 oben) und macht die 3-Klassen-Obergrenze gegenstandslos für die einfachere binäre Frage, löst aber das zugrunde liegende Problem schwacher Merkmalsassoziation nicht auf; der Champion erreicht ein mit der Literatur konsistentes macro-F1 (§6), aber Klassifikation mit den öffentlich verfügbaren Merkmalen bleibt grundsätzlich durch dieselbe Datenlücke begrenzt, ob 3-Klassen oder binär.
#
# **Was erreicht wurde:**
# - Systematischer Vergleich aller 10 Kandidaten aus dem A³-Suchlauf mit ROC/PR/Konfusionsmatrix für den Champion (§1).
# - Fehleranalyse nach sechs Slice-Dimensionen (Unfalltyp, OSM-Straßenklasse, Straßenzustand, Lichtverhältnisse, Niederschlag, Tageszeit): höchste False-Negative-Raten bei Unfallart "Zzs. ruhendes Fz." (`UART=1`, 89,0 %), "Zzs. seitlich gleichfahrendes Fz." (`UART=3`, 74,1 %) und "Zzs. vorausfahrendes Fz." / Auffahrunfall (`UART=2`, 71,5 %), außerdem erhöht im morgendlichen Berufsverkehr (7–9 Uhr, 56,7–60,7 %); Fehler konzentrieren sich systematisch dort, wo die Merkmale am schwächsten trennen, während Niederschlag keinen messbaren Effekt zeigt (§2).
# - Formale Gate-Validierung: **bestanden** gegen beide Q-Phase-Kriterien (macro-F1 0,6039 ≥ 0,55; Recall(KSI) 0,5151 ≥ 0,50) (§3).
# - Gewichtete qualitative Bewertungsmatrix (macro-F1 0,600 vs. 0,518 vs. 0,500), die die Champion-Entscheidung gegenüber den Recall-stärkeren Runner-ups (xgboost, lightgbm) begründet; inklusive eines empirisch geprüften, für alle drei Familien identischen Robustheits-Scores gegenüber fehlenden OSM/DWD-Features (§4).
# - SHAP-Erklärbarkeit (global + 4 Fallbeispiele), konsistent mit der A³-Feature-Evidenz (§5).
# - Literaturabgleich: Test-2024 macro-F1 im zitierten Literaturbereich (Santos 2022, Pakgohar 2021, Schlößler 2024) (§6).
# - Ehrliche Limitationsdiskussion (§7).
# - Vollständiges K-Phase-Artefaktpaket: Pipeline, Schwellenwert, Inference-Contract mit 30 erforderlichen Eingabespalten inklusive Wertebereich/Kategorien und Herkunft je Spalte (§9).
#
# **Ausblick:** Die K-Phase implementiert die Streamlit-App (`app/streamlit_app.py`) gegen `data/processed/c_phase_inference_contract.json` und `data/processed/a3_binary_best_model.joblib`.
#
# **Limitationen (siehe §7):** Selektionsbias, fehlende physische Determinanten, Korrelation ≠ Kausalität, nicht-versionierte OSM-Features, begrenzte geografische/zeitliche Abdeckung, Schwellenwert-Sensitivität, bewusster Recall/macro-F1-Tradeoff.
