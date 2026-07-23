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
# # Unfallatlas Deutschland — U-Phase
#
# **Phase:** Understanding the Data (U) · 2 of 5 · QUA³CK
# **Goal of this notebook:** verify the Q-phase assumptions against the data,
# audit the dataset for quality and leakage, characterise its distributions
# and patterns, and produce a column-by-column preprocessing decision table
# that A³ can implement against.
#
# **Strict scope.** This notebook *observes*, *audits*, and *decides*. It does
# not encode, scale, engineer features, or train models. All transformations
# are A³ work; their selection rationale is documented here.
#
# > Every claim in this notebook must be backed by a cell output above it.
# > Every preprocessing decision in this notebook must appear in the §10
# > decision table as a contract to A³.
#
# ---

# %% [markdown]
# ## Position in the QUA³CK process
#
# | Phase | Notebook | Status |
# |:---|:---|:---:|
# | Q — Question | `01_Q_Phase.ipynb` | ✓ |
# | **U — Understanding** | `02_U_Phase.ipynb` | **→ here** |
# | A³ — Algorithm / Adapt / Adjust | `03_A3_Phase.ipynb` | pending |
# | C — Conclude & Compare | `04_C_Phase.ipynb` | pending |
# | K — Knowledge Transfer | `app/streamlit_app.py` | pending |
#
# ---

# %% [markdown]
# ## 0 — Setup and reproducibility
#
# A U-phase notebook is reproducible or it is not a U-phase notebook. The
# following cells pin versions, hash the data file, configure Plotly, and set
# a deterministic random seed for sampling.
#
# **Plotly is the only plotting library used here** — it is already pinned in
# `pyproject.toml` (≥ 5.22) and renders interactively in Jupyter and VS Code.
# Figures save as standalone `.html` files (preserve interactivity) and
# optionally as `.png` if `kaleido` is installed (`uv pip install kaleido`).

# %%
# Standard library
import hashlib
import subprocess
from collections import Counter
from datetime import datetime
from math import log2
from pathlib import Path

# Third-party — all in pyproject.toml
import duckdb
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Plotly defaults — clean white template, sensible figure size, mobile-friendly
pio.templates.default = "plotly_white"
pio.renderers.default = "plotly_mimetype"
DEFAULT_FIG_W, DEFAULT_FIG_H = 900, 480

# Pandas display
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
np.random.seed(42)

# Project colour palette — kept consistent across all U-phase plots
COLOR_PRIMARY = "#315f7d"  # brand steel blue (matches presentation UI accent)
COLOR_FATAL = "#c1393a"  # class 1 — Getötet
COLOR_SERIOUS = "#e09f3e"  # class 2 — Schwer
COLOR_MINOR = "#5a8db8"  # class 3 — Leicht
COLOURS_SEV = [COLOR_FATAL, COLOR_SERIOUS, COLOR_MINOR]  # ordered 1, 2, 3

# %%
# Human-readable English labels for coded categoricals.
# Raw codes and their definitions follow the Unfallatlas dataset documentation.
ULICHTVERH_LABELS = {0: "Daylight", 1: "Twilight", 2: "Darkness"}
STRZUSTAND_LABELS = {0: "Dry", 1: "Wet/damp/slippery", 2: "Wintry"}
UART_LABELS = {
    0: "Other accident",
    1: "Collision with stationary vehicle",
    2: "Collision with vehicle ahead",
    3: "Collision with vehicle travelling alongside",
    4: "Collision with oncoming vehicle",
    5: "Collision while turning or crossing",
    6: "Collision with pedestrian",
    7: "Impact with road obstacle",
    8: "Departure from road to the right",
    9: "Departure from road to the left",
}
UTYP1_LABELS = {
    1: "Loss-of-control accident",
    2: "Turning accident",
    3: "Crossing or entering accident",
    4: "Pedestrian crossing accident",
    5: "Stationary-traffic accident",
    6: "Longitudinal-traffic accident",
    7: "Other accident",
}
MODE_LABELS = {
    "IstRad": "Bicycle",
    "IstPKW": "Car",
    "IstFuss": "Pedestrian",
    "IstKrad": "Motorcycle",
    "IstGkfz": "Heavy-goods vehicle",
    "IstSonstig": "Other vehicle",
}
DWD_COL_LABELS = {
    "dwd_temp_air_2m": "Air temperature (°C)",
    "dwd_precip_mm": "Precipitation (mm)",
    "dwd_visibility_m": "Visibility (m)",
    "dwd_wind_speed_ms": "Wind speed (m/s)",
    "dwd_station_dist_km": "Distance to DWD station (km)",
    "_precip_bucket": "Precipitation",
}
FEATURE_LABELS = {
    "UKATGEORIE": "Accident severity (target)",
    "UART": "Accident type",
    "UTYP1": "Accident category",
    "ULICHTVERH": "Lighting conditions",
    "STRZUSTAND": "Road condition",
    "UWOCHENTAG": "Weekday",
    "UMONAT": "Month",
    "IstRad": "Bicycle",
    "IstPKW": "Car",
    "IstFuss": "Pedestrian",
    "IstKrad": "Motorcycle",
    "IstGkfz": "Heavy-goods vehicle",
    "IstSonstig": "Other vehicle",
    **DWD_COL_LABELS,
}
COL_CODE_LABELS = {
    "ULICHTVERH": ULICHTVERH_LABELS,
    "STRZUSTAND": STRZUSTAND_LABELS,
    "UART": UART_LABELS,
    "UTYP1": UTYP1_LABELS,
}

# %%
# Paths — robust whether launched from project root or from notebooks/
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = BASE_DIR / "data" / "accidents.parquet"
FIG_DIR = BASE_DIR / "reports" / "figures" / "u_phase"
FIG_DIR.mkdir(parents=True, exist_ok=True)

assert DATA.exists(), (
    f"Data file not found at {DATA}\n"
    "Re-run the consolidation script in src/unfallatlas/data/ to materialise it."
)


def save_fig(fig: go.Figure, slug: str) -> tuple[Path, Path | None]:
    """Persist a Plotly figure as interactive HTML and optionally as static PNG.

    HTML always saves; PNG requires kaleido (uv pip install kaleido).
    Returns (html_path, png_path_or_None).
    """
    html_path = FIG_DIR / f"{slug}.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    png_path = None
    try:
        png_path = FIG_DIR / f"{slug}.png"
        fig.write_image(png_path, scale=2, width=DEFAULT_FIG_W, height=DEFAULT_FIG_H)
    except (ValueError, ImportError, Exception):
        png_path = None
    return html_path, png_path


# %%
# Provenance — recorded once, at the top, for the run.
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


provenance = {
    "source": "GovData / Mobilithek — Unfallatlas Deutschland",
    "licence": "Datenlizenz Deutschland 2.0 (Namensnennung)",
    "file": str(DATA.relative_to(BASE_DIR)),
    "size_mb": round(DATA.stat().st_size / 1_048_576, 2),
    "sha256_16": hashlib.sha256(DATA.read_bytes()).hexdigest()[:16],
    "duckdb": duckdb.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "plotly": plotly.__version__,
    "git_commit": _git_short_sha(),
    "run_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "random_seed": 42,
}
for k, v in provenance.items():
    print(f"  {k:14s} {v}")

# %%
# DuckDB connection — used for any aggregation over the full ~2 M-row table.
con = duckdb.connect()
con.execute("SET memory_limit = '4GB';")

# %% [markdown]
# ## 1 — Schema and dtype audit
#
# The first quantitative output: what columns exist, what are their dtypes, and
# how many distinct values does each one carry?

# %%
schema = con.execute(f"DESCRIBE SELECT * FROM '{DATA}'").df()
print(f"columns: {len(schema)}")
schema


# %%
# Cardinality + missingness — the audit one-pager.
def audit_table(parquet_path: Path) -> pd.DataFrame:
    cols = con.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'").df()["column_name"]
    rows = []
    for col in cols:
        q = f"""
            SELECT
                COUNT(*) AS n,
                COUNT(DISTINCT "{col}") AS n_unique,
                SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS n_missing
            FROM '{parquet_path}'
        """
        r = con.execute(q).fetchone()
        rows.append(
            {
                "column": col,
                "n_unique": r[1],
                "n_missing": r[2],
                "pct_missing": 100.0 * r[2] / r[0],
            }
        )
    return pd.DataFrame(rows)


audit = audit_table(DATA)
audit

# %% [markdown]
# ### Semantic-type annotation
#
# Storage dtype is not modelling type. The table below records, for each column,
# how A³ should treat it. This annotation is the most useful single artefact U
# hands to A³.
#
# | Column | Storage | Semantic type | Treatment hint for A³ |
# |:---|:---|:---|:---|
# | `OBJECTID` | INT | identifier | drop before modelling |
# | `UJAHR` | SMALLINT | temporal ordinal | only used for the chronological split, not as a feature |
# | `UMONAT` | TINYINT | cyclic ordinal (period 12) | sin/cos encoding in A³ |
# | `USTUNDE` | TINYINT | cyclic ordinal (period 24) | sin/cos encoding in A³ |
# | `UWOCHENTAG` | TINYINT | cyclic ordinal (period 7) | sin/cos encoding in A³ |
# | **`UKATGEORIE`** | TINYINT | **ordinal target** | label — never a feature |
# | `UART` | TINYINT | nominal categorical | one-hot or target-encoded; **leakage probe required** |
# | `UTYP1` | TINYINT | nominal categorical | one-hot or target-encoded; **leakage probe required** |
# | `ULICHTVERH` | TINYINT | nominal categorical (3 levels) | one-hot |
# | `STRZUSTAND` | TINYINT | nominal categorical (3 levels) | one-hot |
# | `IstRad` … `IstSonstig` | BOOLEAN | binary | pass-through |
# | `LON`, `LAT` | DOUBLE | continuous spatial | `StandardScaler` only if a distance-based model is added; tree models pass-through |
# | `UREGBEZ` | VARCHAR | nominal categorical | target-encoding |
# | `UKREIS` | VARCHAR | high-cardinality nominal | target-encoding; one-hot would explode the matrix |
# | `UGEMEINDE` | VARCHAR | very-high-cardinality nominal | drop or hash-encode |
#
# > **Note.** `UART` and `UTYP1` are flagged for leakage probing in §9 because
# > their semantic relationship to the severity outcome is not obvious from
# > the documentation alone.
#
# ---

# %% [markdown]
# ## 2 — Volume and temporal coverage
#
# Verifies the Q-phase assumption: ~2.09 M rows, 9 vintages 2016 – 2024.

# %%
total = con.execute(f"""
    SELECT COUNT(*) AS n_rows,
           COUNT(DISTINCT UJAHR) AS n_years,
           MIN(UJAHR) AS first_year,
           MAX(UJAHR) AS last_year
    FROM '{DATA}'
""").df()
total

# %%
by_year = con.execute(f"""
    SELECT UJAHR AS year,
           COUNT(*) AS n_rows,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM '{DATA}'
    GROUP BY UJAHR
    ORDER BY UJAHR
""").df()
by_year

# %%
fig = px.bar(
    by_year,
    x="year",
    y="n_rows",
    text="n_rows",
    title="Accidents per year — Unfallatlas 2016 – 2024",
    labels={"year": "", "n_rows": "accidents"},
    color_discrete_sequence=[COLOR_PRIMARY],
    height=DEFAULT_FIG_H,
)
fig.update_traces(
    texttemplate="%{text:,}",
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>%{y:,} accidents<extra></extra>",
)
fig.update_layout(xaxis=dict(tickmode="linear"), showlegend=False)
save_fig(fig, "02_rows_per_year")
fig.show()

# %% [markdown]
# > **Observation.** All nine years are present. The visible 2020 dip is
# > consistent with COVID-19 mobility reduction and matches BASt yearly
# > reports — structural, not a data-quality issue. The level recovers
# > steadily from 2021 onwards; by 2023 it matches and marginally exceeds
# > the 2019 pre-pandemic peak (269,048 vs 268,370). A³ should note that
# > the 2024 test year (268,519) sits in the same volume regime as the
# > 2019 peak, not the lower 2016 – 2018 range.
#
# ---

# %% [markdown]
# ## 3 — Data quality audit
#
# Five sub-audits: missingness, sentinel values, duplicates, range checks,
# and consistency rules.

# %% [markdown]
# ### 3.1  Missing values

# %%
missing = audit[audit["n_missing"] > 0].sort_values("pct_missing", ascending=False)
print(f"columns with missing values: {len(missing)} / {len(audit)}")
missing

# %%
# Missingness map — fully native Plotly (no missingno dependency).
SAMPLE_N = 20_000
df_sample = con.execute(f"SELECT * FROM '{DATA}' USING SAMPLE {SAMPLE_N} ROWS (reservoir, 42)").df()

miss_matrix = df_sample.isna().astype(int).T  # rows = columns, cols = sample rows
fig = px.imshow(
    miss_matrix.values,
    aspect="auto",
    color_continuous_scale=[[0, "#f4f4f4"], [1, "#1a1a1a"]],
    y=miss_matrix.index.tolist(),
    labels={"x": "row index in sample", "y": "column", "color": "missing"},
    title=f"Missingness map — dark = NaN (n = {SAMPLE_N:,} sampled rows)",
    height=520,
)
fig.update_xaxes(showticklabels=False)
fig.update_coloraxes(showscale=False)
save_fig(fig, "03_missingness_matrix")
fig.show()

# %% [markdown]
# > **Observation.** Missingness is almost entirely confined to `IstGkfz`
# > (heavy-goods vehicle flag); all other 20 columns are fully populated in
# > the sample. The sparse missing pattern is consistent with rare vehicle
# > types being underreported in the police record.
# >
# > **Decision.** Per-column missing-value strategies are recorded in the §10
# > decision table. Imputation is *not* performed here — that is A³ work
# > inside a `Pipeline` so statistics do not leak across splits.

# %% [markdown]
# ### 3.2  Sentinel-value scan
#
# Sentinels (e.g. `-1`, `9999`, empty strings) are not caught by `isna()` and
# must be hunted explicitly. The Unfallatlas codebook uses small positive
# integers, so any negative numeric value is suspicious.

# %%
# Min / max of every numeric column.
numeric_cols = [
    "UJAHR",
    "UMONAT",
    "USTUNDE",
    "UWOCHENTAG",
    "UKATGEORIE",
    "UART",
    "UTYP1",
    "ULICHTVERH",
    "STRZUSTAND",
    "LON",
    "LAT",
]
minmax = (
    con.execute(f"""
    SELECT {", ".join(f"MIN({c}) AS min_{c}, MAX({c}) AS max_{c}" for c in numeric_cols)}
    FROM '{DATA}'
""")
    .df()
    .T
)
minmax.columns = ["value"]
minmax

# %%
# String-column sentinel-like values.
str_cols = ["UREGBEZ", "UKREIS", "UGEMEINDE"]
suspicious_set = {"unknown", "na", "n/a", "", "null", "none", "?", "-"}
for col in str_cols:
    df_col = con.execute(f"SELECT DISTINCT \"{col}\" AS v FROM '{DATA}'").df()["v"].astype(str)
    suspicious = df_col[df_col.str.lower().isin(suspicious_set)]
    print(f"  {col:10s}  distinct = {len(df_col):>6}  suspicious = {len(suspicious)}")

# %% [markdown]
# ### 3.3  Duplicates

# %%
dupe_ids = con.execute(f"""
    SELECT OBJECTID, COUNT(*) AS n
    FROM '{DATA}'
    GROUP BY OBJECTID
    HAVING COUNT(*) > 1
    LIMIT 10
""").df()
print(f"duplicate OBJECTIDs: {len(dupe_ids):,}")
if len(dupe_ids):
    display(dupe_ids)

# %%
# Exact row duplicates across all 21 columns.
n_total = con.execute(f"SELECT COUNT(*) FROM '{DATA}'").fetchone()[0]
n_distinct = con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM '{DATA}')").fetchone()[0]
print(f"  total rows          : {n_total:,}")
print(f"  distinct rows       : {n_distinct:,}")
print(f"  exact row duplicates: {n_total - n_distinct:,}")

# %% [markdown]
# ### 3.4  Range checks
#
# Domain-bound checks catch errors that statistical outlier rules will miss.

# %%
DE_BBOX = {"lat_min": 47.27, "lat_max": 55.06, "lon_min": 5.87, "lon_max": 15.04}

range_checks = (
    con.execute(f"""
    SELECT
        SUM(CASE WHEN LAT NOT BETWEEN {DE_BBOX["lat_min"]} AND {DE_BBOX["lat_max"]}
                 THEN 1 ELSE 0 END) AS lat_outside_de,
        SUM(CASE WHEN LON NOT BETWEEN {DE_BBOX["lon_min"]} AND {DE_BBOX["lon_max"]}
                 THEN 1 ELSE 0 END) AS lon_outside_de,
        SUM(CASE WHEN USTUNDE    NOT BETWEEN 0 AND 23 THEN 1 ELSE 0 END) AS hour_invalid,
        SUM(CASE WHEN UMONAT     NOT BETWEEN 1 AND 12 THEN 1 ELSE 0 END) AS month_invalid,
        SUM(CASE WHEN UWOCHENTAG NOT BETWEEN 1 AND 7  THEN 1 ELSE 0 END) AS weekday_invalid,
        SUM(CASE WHEN UKATGEORIE NOT IN (1, 2, 3)     THEN 1 ELSE 0 END) AS severity_invalid
    FROM '{DATA}'
""")
    .df()
    .T
)
range_checks.columns = ["count"]
range_checks

# %% [markdown]
# ### 3.5  Consistency rules
#
# Domain rule: every accident must involve at least one transport mode.

# %%
mode_cols = ["IstRad", "IstPKW", "IstFuss", "IstKrad", "IstGkfz", "IstSonstig"]
violation = con.execute(f"""
    SELECT COUNT(*) AS n_no_mode
    FROM '{DATA}'
    WHERE CAST(IstRad AS INT) + CAST(IstPKW AS INT) + CAST(IstFuss AS INT)
        + CAST(IstKrad AS INT) + CAST(IstGkfz AS INT) + CAST(IstSonstig AS INT) = 0
""").df()
print("rows with no transport mode flagged:", int(violation.iloc[0, 0]))

# %% [markdown]
# > **Quality audit summary.** Document each finding above in the §11 risk
# > list if it affects A³. Range failures and consistency violations above
# > 0.1 % of rows warrant a return to Q for re-discussion.
#
# ---

# %% [markdown]
# ## 4 — Target variable
#
# The audit begins with the original three-class target: fatal, serious injury,
# and slight injury. Keeping the original classes visible is essential because
# their imbalance and separability determine whether that formulation is viable.

# %% [markdown]
# > **Staged target policy.** The plots in sections 4–8 retain all three source
# > classes. The operational KSI fallback is introduced only after this evidence
# > is assembled, in section 10. This keeps the reformulation traceable: stable
# > class shares support temporal evaluation, while imbalance, weak associations,
# > and missing physical determinants challenge fatal-class separation.

# %%
target = con.execute(f"""
    SELECT UKATGEORIE AS class,
           CASE UKATGEORIE WHEN 1 THEN '1 — Getötet'
                          WHEN 2 THEN '2 — Schwer verletzt'
                          ELSE     '3 — Leicht verletzt' END AS label,
           COUNT(*) AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM '{DATA}'
    GROUP BY UKATGEORIE
    ORDER BY UKATGEORIE
""").df()
target

# %%
n_max = target["n"].max()
n_min = target["n"].min()
print(f"imbalance ratio (majority : minority) = {n_max / n_min:.1f} : 1")

# %%
# Horizontal stacked bar — single bar, three segments, percentages annotated.
fig = go.Figure()
left = 0
for (_, row), color in zip(target.iterrows(), COLOURS_SEV):
    fig.add_trace(
        go.Bar(
            x=[row["pct"]],
            y=["share"],
            orientation="h",
            name=row["label"],
            marker_color=color,
            marker_line_color="white",
            marker_line_width=2,
            text=f"{row['pct']:.1f}%",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=14),
            hovertemplate=f"<b>{row['label']}</b><br>n = {row['n']:,}<br>%{{x:.2f}}%<extra></extra>",
        )
    )
    left += row["pct"]
fig.update_layout(
    title="UKATGEORIE — class distribution (full dataset)",
    barmode="stack",
    height=240,
    showlegend=True,
    legend=dict(orientation="h", y=-0.4, x=0.5, xanchor="center"),
    xaxis=dict(range=[0, 100], title="share (%)"),
    yaxis=dict(showticklabels=False),
    margin=dict(l=20, r=20, t=60, b=80),
)
save_fig(fig, "04_target_distribution_stacked")
fig.show()

# %% [markdown]
# ### 4.2  Stability across years
#
# A model trained on 2016 – 2022 will be evaluated on 2024. If class
# proportions drift across years, the held-out performance estimate is biased.

# %%
target_by_year = con.execute(f"""
    SELECT UJAHR, UKATGEORIE, COUNT(*) AS n
    FROM '{DATA}'
    GROUP BY UJAHR, UKATGEORIE
    ORDER BY UJAHR, UKATGEORIE
""").df()
pivot = target_by_year.pivot(index="UJAHR", columns="UKATGEORIE", values="n")
pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pct.columns = ["1 — Getötet", "2 — Schwer", "3 — Leicht"]
pct.round(2)

# %%
pct_long = pct.reset_index().melt(id_vars="UJAHR", var_name="class", value_name="pct")
fig = px.line(
    pct_long,
    x="UJAHR",
    y="pct",
    color="class",
    markers=True,
    title="Class proportions by year — stability check",
    labels={"UJAHR": "", "pct": "share (%)", "class": "severity"},
    color_discrete_sequence=COLOURS_SEV,
    height=DEFAULT_FIG_H,
)
fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>")
fig.update_layout(xaxis=dict(tickmode="linear"))
save_fig(fig, "04_target_stability_by_year")
fig.show()

# %% [markdown]
# > **Observation.** Class shares are stable to within ~1 pp across the nine
# > years — no structural drift that would invalidate the chronological split.
#
# ---

# %% [markdown]
# ## 5 — Univariate distributions
#
# A first look at each feature in isolation. Categorical features get
# countplots; binary features a grid of counts; continuous features histograms.

# %%
# Categorical features — 2 × 2 grid of countplots.
cat_cols = ["ULICHTVERH", "STRZUSTAND", "UART", "UTYP1"]
counts = {
    col: con.execute(
        f"SELECT {col} AS v, COUNT(*) AS n FROM '{DATA}' GROUP BY {col} ORDER BY {col}"
    ).df()
    for col in cat_cols
}

subplot_titles = [FEATURE_LABELS.get(c, c) for c in cat_cols]
fig = make_subplots(
    rows=2, cols=2, subplot_titles=subplot_titles, vertical_spacing=0.18, horizontal_spacing=0.10
)
positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
for col, (r, c) in zip(cat_cols, positions):
    d = counts[col].copy()
    lbl_map = COL_CODE_LABELS.get(col, {})
    d["label"] = d["v"].map(lbl_map).fillna(d["v"].astype(str))
    fig.add_trace(
        go.Bar(
            x=d["label"],
            y=d["n"],
            marker_color=COLOR_PRIMARY,
            hovertemplate=(
                f"<b>{FEATURE_LABELS.get(col, col)}</b>: %{{x}}<br>n = %{{y:,}}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=r,
        col=c,
    )
fig.update_layout(
    title="Categorical feature distributions",
    height=620,
    margin=dict(t=80),
)
save_fig(fig, "05_categorical_grid")
fig.show()

# %% [markdown]
# **Interpretation:** Daylight (ULICHTVERH=0) and dry road (STRZUSTAND=0) dominate as expected. For accident type (UART), collisions with merging/crossing vehicles (code 5) are most frequent, while road-departure accidents (codes 8/9) carry the highest fatality rate. "Other type" (code 0) accounts for 15 % of records and has an unusual profile: 58 % bicycle involvement.

# %%
mode_counts = (
    con.execute(f"""
    SELECT 'IstRad'     AS mode, SUM(CAST(IstRad     AS INT)) AS n FROM '{DATA}' UNION ALL
    SELECT 'IstPKW'     AS mode, SUM(CAST(IstPKW     AS INT)) AS n FROM '{DATA}' UNION ALL
    SELECT 'IstFuss'    AS mode, SUM(CAST(IstFuss    AS INT)) AS n FROM '{DATA}' UNION ALL
    SELECT 'IstKrad'    AS mode, SUM(CAST(IstKrad    AS INT)) AS n FROM '{DATA}' UNION ALL
    SELECT 'IstGkfz'    AS mode, SUM(CAST(IstGkfz    AS INT)) AS n FROM '{DATA}' UNION ALL
    SELECT 'IstSonstig' AS mode, SUM(CAST(IstSonstig AS INT)) AS n FROM '{DATA}'
""")
    .df()
    .sort_values("n", ascending=True)
)
mode_counts["mode_label"] = mode_counts["mode"].map(MODE_LABELS)

fig = px.bar(
    mode_counts,
    x="n",
    y="mode_label",
    orientation="h",
    title="Transport-mode involvement — rows with each flag = 1",
    labels={"n": "accidents", "mode_label": "transport mode"},
    color_discrete_sequence=[COLOR_PRIMARY],
    height=380,
)
fig.update_traces(
    hovertemplate="<b>%{y}</b><br>%{x:,} accidents<extra></extra>",
    texttemplate="%{x:,}",
    textposition="outside",
)
save_fig(fig, "05_transport_mode_counts")
fig.show()

# %% [markdown]
# **Interpretation:** Car involvement (IstPKW) dominates at ~77 % of accidents. The high bicycle share (~30 %) is notable and relevant to the severity model, since cyclist accidents have a distinct injury profile. Heavy-goods vehicles (IstGkfz) are rare (~4 %) but associated with higher fatality rates. All transport-mode flags enter the model as binary features.

# %%
# Geographic coordinate histograms with skew annotation.
coords = con.execute(f"SELECT LON, LAT FROM '{DATA}'").df()
skew_lon = float(coords["LON"].skew())
skew_lat = float(coords["LAT"].skew())

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(f"LON — skew = {skew_lon:.2f}", f"LAT — skew = {skew_lat:.2f}"),
    horizontal_spacing=0.10,
)
fig.add_trace(
    go.Histogram(
        x=coords["LON"], nbinsx=60, marker_color=COLOR_PRIMARY, name="LON", showlegend=False
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Histogram(
        x=coords["LAT"], nbinsx=60, marker_color=COLOR_PRIMARY, name="LAT", showlegend=False
    ),
    row=1,
    col=2,
)
fig.update_layout(title="Coordinate distributions", height=380)
save_fig(fig, "05_coords_histograms")
fig.show()


# %% [markdown]
# > **Observation.** LON (skew = 0.30) and LAT (skew = 0.10) reflect
# > Germany's centre-of-mass distribution — neither is skewed enough to
# > demand a transformation. Tree models will use these directly; if a
# > distance-based baseline is added in A³, `StandardScaler` is appropriate.
#
# ---

# %% [markdown]
# ## 6 — Bivariate analysis — feature × target
#
# Pearson correlation on coded categoricals is meaningless. The correct
# measure for categorical-vs-categorical association is **Cramér's V**
# (range 0 … 1).


# %%
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma & Wicher, 2013)."""
    from scipy.stats import chi2_contingency

    confusion = pd.crosstab(x, y)
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
    return float("nan") if denom <= 0 else np.sqrt(phi2c / denom)


CV_SAMPLE = 100_000
df_cv = con.execute(f"SELECT * FROM '{DATA}' USING SAMPLE {CV_SAMPLE} ROWS (reservoir, 42)").df()

cv_cols = [
    "UKATGEORIE",
    "UART",
    "UTYP1",
    "ULICHTVERH",
    "STRZUSTAND",
    "UWOCHENTAG",
    "UMONAT",
    "IstRad",
    "IstPKW",
    "IstFuss",
    "IstKrad",
    "IstGkfz",
    "IstSonstig",
]

cv_matrix = pd.DataFrame(index=cv_cols, columns=cv_cols, dtype=float)
for a in cv_cols:
    for b in cv_cols:
        cv_matrix.loc[a, b] = cramers_v(df_cv[a], df_cv[b])
cv_matrix.round(3)

# %%
cv_labeled = cv_matrix.copy()
cv_labeled.index = [FEATURE_LABELS.get(c, c) for c in cv_matrix.index]
cv_labeled.columns = [FEATURE_LABELS.get(c, c) for c in cv_matrix.columns]

fig = px.imshow(
    cv_labeled.astype(float).values,
    x=cv_labeled.columns,
    y=cv_labeled.index,
    color_continuous_scale="Plasma",
    zmin=0,
    zmax=1,
    text_auto=".2f",
    aspect="equal",
    title=f"Cramér's V — categorical association (n = {CV_SAMPLE:,} sample)",
    labels=dict(color="Cramér's V"),
    height=640,
)
fig.update_traces(
    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Cramér's V = %{z:.3f}<extra></extra>",
)
fig.update_xaxes(tickangle=-45)
save_fig(fig, "06_cramers_v_heatmap")
fig.show()

# %% [markdown]
# > **Reading guide.** The first row / column (`UKATGEORIE`) shows each
# > feature's association with the target. In this dataset all target-row
# > values stay below 0.15 (highest: Unfallart = 0.13, Unfalltyp = 0.11)
# > — no single feature shows a definitional association. The high
# > off-diagonal values (e.g., Unfallart–Fußgänger = 0.93, Unfalltyp–PKW
# > = 0.43) reveal inter-feature collinearity; the conditional-entropy
# > probe in §9 tests whether those structural correlations introduce leakage.

# %%
# Severity-share decomposition by lighting and road condition.
fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=(
        f"Severity share by {FEATURE_LABELS['ULICHTVERH']}",
        f"Severity share by {FEATURE_LABELS['STRZUSTAND']}",
    ),
    horizontal_spacing=0.10,
)
labels_sev = ["1 Fatal", "2 Serious injury", "3 Slight injury"]
for col_i, col in enumerate(["ULICHTVERH", "STRZUSTAND"], start=1):
    lbl_map = COL_CODE_LABELS[col]
    pct_df = con.execute(f"""
        SELECT {col} AS x, UKATGEORIE AS sev, COUNT(*) AS n
        FROM '{DATA}' GROUP BY {col}, UKATGEORIE
    """).df()
    pivot = pct_df.pivot(index="x", columns="sev", values="n").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
    # Replace numeric index with readable labels
    pivot.index = [lbl_map.get(i, str(i)) for i in pivot.index]
    for sev_i, (sev, color, label) in enumerate(zip([1, 2, 3], COLOURS_SEV, labels_sev)):
        if sev not in pivot.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=pivot.index.tolist(),
                y=pivot[sev],
                name=label,
                marker_color=color,
                showlegend=(col_i == 1),
                hovertemplate=(f"<b>%{{x}}</b><br>{label}: %{{y:.2f}}%<extra></extra>"),
            ),
            row=1,
            col=col_i,
        )
fig.update_layout(
    barmode="stack",
    title="Severity decomposition by environment",
    height=460,
    legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
)
fig.update_yaxes(title="share (%)", col=1)
save_fig(fig, "06_severity_by_conditions")
fig.show()

# %% [markdown]
# > **Observation.** Severity shares are near-uniform across both
# > lighting and road-condition categories — all three Lichtverhältnisse
# > and all three Straßenzustand groups show ≈ 80 % Leicht / 18 % Schwer
# > / 1–2 % Getötet. This is consistent with the low Cramér's V scores
# > (0.02 and 0.01 respectively): these features do not strongly predict
# > severity alone. Signal is expected through interactions (e.g.,
# > Dunkelheit + Winterglatt) rather than as standalone predictors.

# %% [markdown]
# ---

# %% [markdown]
# ## 7 — Temporal patterns
#
# Hourly profile, weekday × hour heatmaps, and stability checks.

# %%
hourly = con.execute(f"""
    SELECT USTUNDE AS hour,
           COUNT(*) AS n,
           AVG(CAST(UKATGEORIE AS DOUBLE)) AS mean_severity
    FROM '{DATA}'
    GROUP BY USTUNDE
    ORDER BY USTUNDE
""").df()

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(
        x=hourly["hour"],
        y=hourly["n"],
        marker_color=COLOR_PRIMARY,
        opacity=0.85,
        name="Accidents (count)",
        hovertemplate="Hour %{x}<br>%{y:,} accidents<extra></extra>",
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=hourly["hour"],
        y=hourly["mean_severity"],
        mode="lines+markers",
        marker=dict(color=COLOR_FATAL, size=8),
        line=dict(color=COLOR_FATAL, width=2),
        name="Mean severity code (1=fatal, 3=slight)",
        hovertemplate="Hour %{x}<br>Mean severity code = %{y:.3f}<extra></extra>",
    ),
    secondary_y=True,
)
fig.update_layout(
    title="Hourly profile — frequency and mean severity",
    height=DEFAULT_FIG_H,
    legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
)
fig.update_xaxes(title="Hour", tickmode="linear", dtick=1)
fig.update_yaxes(title_text="Accidents", secondary_y=False)
fig.update_yaxes(title_text="Mean severity code (lower = more severe)", secondary_y=True)
save_fig(fig, "07_hourly_profile")
fig.show()

# %% [markdown]
# **Interpretation:** Accident frequency peaks during the afternoon commute
# (15–17 h), with a secondary morning peak (7–9 h). Night-time accidents
# (0–5 h) are rarer but show lower mean Unfallkategorie (≈ 2.73 vs ≈ 2.82
# during daylight hours) — i.e., more severe on average — driven by higher
# speeds and lower visibility. USTUNDE is encoded as cyclic (sin/cos) features
# for modelling.

# %%
# Weekday × hour heatmaps — count + mean severity.
wh = con.execute(f"""
    SELECT UWOCHENTAG AS weekday, USTUNDE AS hour,
           COUNT(*) AS n,
           AVG(CAST(UKATGEORIE AS DOUBLE)) AS mean_severity
    FROM '{DATA}'
    GROUP BY UWOCHENTAG, USTUNDE
""").df()

# Weekday coding: 1=Sun, 2=Mon … 7=Sat. Reorder Mon-first.
weekday_names = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}
order = [2, 3, 4, 5, 6, 7, 1]

count_pivot = wh.pivot(index="weekday", columns="hour", values="n").reindex(order)
sev_pivot = wh.pivot(index="weekday", columns="hour", values="mean_severity").reindex(order)
y_labels = [weekday_names[i] for i in order]
x_labels = list(count_pivot.columns)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "Weekday × hour — accident count",
        "Weekday × hour — mean severity code (darker = more severe)",
    ),
    vertical_spacing=0.18,
)
fig.add_trace(
    go.Heatmap(
        z=count_pivot.values,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        colorbar=dict(title="Count", y=0.78, len=0.42),
        hovertemplate="%{y} %{x}:00<br>%{z:,} accidents<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Heatmap(
        z=sev_pivot.values,
        x=x_labels,
        y=y_labels,
        colorscale="Reds_r",
        colorbar=dict(title="Mean severity<br>(1=fatal)", y=0.22, len=0.42),
        hovertemplate="%{y} %{x}:00<br>Mean severity = %{z:.3f}<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.update_layout(title="Weekday × hour patterns", height=720)
fig.update_xaxes(title_text="Hour", row=2, col=1)
save_fig(fig, "07_weekday_hour_heatmaps")
fig.show()

# %% [markdown]
# > **Observation.** Frequency peaks during commuter hours (7 – 9 and
# > 15 – 18) on weekdays; severity is *higher* (lower mean) at night and on
# > weekends — the two plots tell different stories, which is why both must
# > be shown. A model that uses only `USTUNDE` sees the count signal; a model
# > that uses `USTUNDE` and `UWOCHENTAG` together can resolve the interaction.
#
# ---

# %% [markdown]
# ## 8 — Spatial patterns
#
# Two views: an interactive density map on a representative sample, and a
# Bundesland-level aggregate of fatal-accident share. The Bundesland is
# derived from the first two digits of `UKREIS` (per `AGENTS.md`).

# %%
# Interactive density map (OpenStreetMap, no token required).
SAMPLE_GEO = 50_000
geo_sample = con.execute(
    f"SELECT LON, LAT, UKATGEORIE FROM '{DATA}' USING SAMPLE {SAMPLE_GEO} ROWS (reservoir, 42)"
).df()
geo_sample["severity_label"] = geo_sample["UKATGEORIE"].map(
    {
        1: "1 — Fatal",
        2: "2 — Serious injury",
        3: "3 — Slight injury",
    }
)

fig = px.scatter_mapbox(
    geo_sample,
    lat="LAT",
    lon="LON",
    color="severity_label",
    color_discrete_map={
        "1 — Fatal": COLOR_FATAL,
        "2 — Serious injury": COLOR_SERIOUS,
        "3 — Slight injury": COLOR_MINOR,
    },
    category_orders={"severity_label": ["1 — Fatal", "2 — Serious injury", "3 — Slight injury"]},
    opacity=0.45,
    zoom=5.2,
    center={"lat": 51.2, "lon": 10.4},
    mapbox_style="open-street-map",
    title=f"Accident locations — sample of {SAMPLE_GEO:,} rows",
    height=640,
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(
    legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
    margin=dict(l=0, r=0, t=60, b=20),
)
save_fig(fig, "08_geo_density_map")
fig.show()

# %% [markdown]
# **Interpretation:** Accidents are concentrated in urban agglomerations (Berlin, Ruhr, Munich), but fatal accidents (red) occur disproportionately on rural roads and federal highways where higher speeds and absent median barriers amplify injury outcomes. This urban–rural contrast supports hypothesis H4 (spatial heterogeneity).

# %%
BL_NAMES = {
    1: "Schleswig-Holstein",
    2: "Hamburg",
    3: "Niedersachsen",
    4: "Bremen",
    5: "Nordrhein-Westfalen",
    6: "Hessen",
    7: "Rheinland-Pfalz",
    8: "Baden-Württemberg",
    9: "Bayern",
    10: "Saarland",
    11: "Berlin",
    12: "Brandenburg",
    13: "Mecklenburg-Vorpommern",
    14: "Sachsen",
    15: "Sachsen-Anhalt",
    16: "Thüringen",
}

bl = con.execute(f"""
    SELECT CAST(SUBSTR(UKREIS, 1, 2) AS INT) AS uland,
           COUNT(*) AS n,
           AVG(CAST(UKATGEORIE AS DOUBLE)) AS mean_severity,
           SUM(CASE WHEN UKATGEORIE = 1 THEN 1 ELSE 0 END) AS n_fatal,
           100.0 * SUM(CASE WHEN UKATGEORIE = 1 THEN 1 ELSE 0 END)
                 / COUNT(*) AS pct_fatal
    FROM '{DATA}'
    GROUP BY uland
    ORDER BY uland
""").df()
bl["name"] = bl["uland"].map(BL_NAMES)
bl

# %%
bl_sorted = bl.sort_values("pct_fatal", ascending=True)
fig = px.bar(
    bl_sorted,
    x="pct_fatal",
    y="name",
    orientation="h",
    title="Share of fatal accidents (class 1) by federal state",
    labels={"pct_fatal": "% fatal", "name": ""},
    color_discrete_sequence=[COLOR_FATAL],
    height=560,
    custom_data=["n", "n_fatal"],
)
fig.update_traces(
    hovertemplate=(
        "<b>%{y}</b><br>"
        "%{x:.2f}% fatal<br>"
        "total = %{customdata[0]:,}<br>"
        "fatal = %{customdata[1]:,}<extra></extra>"
    ),
    texttemplate="%{x:.2f}%",
    textposition="outside",
)
fig.update_layout(xaxis=dict(range=[0, bl_sorted["pct_fatal"].max() * 1.35]))
save_fig(fig, "08_pct_fatal_by_bundesland")
fig.show()

# %% [markdown]
# > **Observation.** Thüringen (0.88 %) and Sachsen-Anhalt (0.68 %) lead
# > by a wide margin; eastern Bundesländer and Bayern dominate the top half.
# > Rheinland-Pfalz (0.23 %) and Baden-Württemberg (0.29 %) sit at the low
# > end. The rural / urban narrative holds broadly — city-states Hamburg
# > (0.34 %) and Bremen (0.35 %) are below average — but Berlin (0.43 %,
# > 5th) and Bayern (0.50 %, 3rd) show the split is not clean. Higher speeds
# > and longer rural rescue times likely drive the eastern pattern. The
# > Q-phase scoping note about "no causal claims" applies.
#
# ---

# %% [markdown]
# ## 8.5 — DWD weather enrichment: schema and coverage audit
#
# The DWD join is a precondition for all weather features used in A³. Before
# treating these columns as modelling inputs, three conditions must hold:
#
# 1. **Coverage:** ≥ 95 % of accidents have a DWD station within 30 km.
# 2. **Temporal completeness:** DWD readings present for ≥ 95 % of accident-hours.
# 3. **No systematic gaps** correlated with severity class (audited in §9.4).
#
# Any failure of conditions 1 or 2 requires revisiting the `max_km` parameter
# or adopting an explicit missing-data strategy before A³.

# %%
# DWD enrichment — downloads on first run (~10–30 min depending on bandwidth).
# Subsequent runs read from the Parquet cache in data/raw/dwd/ and data/interim/.

from unfallatlas.data.dwd import build_weather_features  # noqa: E402

DWD_RAW = BASE_DIR / "data" / "raw" / "dwd"
DWD_INTERIM = BASE_DIR / "data" / "interim"

_enriched_cache = DWD_INTERIM / "accidents_with_weather.parquet"

try:
    if _enriched_cache.exists():
        df_weather = pd.read_parquet(_enriched_cache)
    else:
        df_full = pd.read_parquet(DATA)
        df_weather = build_weather_features(
            df_full,
            raw_cache_dir=DWD_RAW,
            interim_cache_dir=DWD_INTERIM,
            max_km=30.0,
        )
    con.register("df_weather", df_weather)
    DWD_COLS = [
        "dwd_temp_air_2m",
        "dwd_precip_mm",
        "dwd_visibility_m",
        "dwd_wind_speed_ms",
        "dwd_station_dist_km",
    ]
    print(f"df_weather shape      : {df_weather.shape}")
    print(f"Weather columns found : {[c for c in DWD_COLS if c in df_weather.columns]}")
except Exception as _dwd_err:
    import warnings

    warnings.warn(
        f"DWD enrichment failed: {_dwd_err}\n"
        "§8.5 – §8.7 and §9.4 require DWD data.\n"
        "Run build_weather_features() manually to fetch it.",
        stacklevel=1,
    )
    df_weather = None
    DWD_COLS = []
    miss_df = pd.DataFrame(columns=["column", "n_missing", "pct_missing"])

# %%
if df_weather is not None:
    n_total = len(df_weather)
    miss_rows = []
    for col in DWD_COLS:
        if col not in df_weather.columns:
            continue
        n_miss = int(df_weather[col].isna().sum())
        miss_rows.append(
            {
                "column": col,
                "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / n_total, 2),
            }
        )
    miss_df = pd.DataFrame(miss_rows)
    print(miss_df.to_string(index=False))

# %%
if df_weather is not None and len(miss_df) > 0:
    miss_plot = miss_df.copy()
    miss_plot["label"] = miss_plot["column"].map(DWD_COL_LABELS).fillna(miss_plot["column"])
    fig = px.bar(
        miss_plot,
        x="pct_missing",
        y="label",
        orientation="h",
        title="DWD weather features — missing value rates",
        labels={"pct_missing": "% missing", "label": ""},
        color_discrete_sequence=[COLOR_PRIMARY],
        text="pct_missing",
        height=340,
    )
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    save_fig(fig, "08_5_dwd_missing_rates")
    fig.show()

# %%
if df_weather is not None and "dwd_station_dist_km" in df_weather.columns:
    dist = df_weather["dwd_station_dist_km"].dropna()
    n_within_30 = int((dist <= 30).sum())
    pct_covered = 100 * n_within_30 / len(dist)
    print(
        f"Accidents with DWD station ≤ 30 km: {n_within_30:,} / {len(dist):,}  ({pct_covered:.1f}%)"
    )

    fig = px.histogram(
        dist.clip(upper=100),
        nbins=60,
        title="Distance to nearest DWD station (km)",
        labels={"value": "distance (km)", "count": "accidents"},
        color_discrete_sequence=[COLOR_PRIMARY],
        height=380,
    )
    fig.add_vline(
        x=30,
        line_dash="dash",
        line_color=COLOR_FATAL,
        annotation_text="30 km threshold",
        annotation_position="top right",
    )
    save_fig(fig, "08_5_dwd_station_distance")
    fig.show()

# %%
if df_weather is not None and "dwd_temp_air_2m" in df_weather.columns:
    temporal_cov = con.execute("""
        SELECT UJAHR AS year,
               COUNT(*) AS n_total,
               SUM(CASE WHEN dwd_temp_air_2m IS NOT NULL THEN 1 ELSE 0 END) AS n_valid,
               ROUND(100.0 * SUM(CASE WHEN dwd_temp_air_2m IS NOT NULL THEN 1 ELSE 0 END)
                     / COUNT(*), 2) AS pct_valid
        FROM df_weather
        GROUP BY UJAHR
        ORDER BY UJAHR
    """).df()
    fig = px.line(
        temporal_cov,
        x="year",
        y="pct_valid",
        markers=True,
        title="DWD temperature coverage (% valid readings) per accident year",
        labels={"year": "", "pct_valid": "% valid"},
        color_discrete_sequence=[COLOR_PRIMARY],
        height=360,
    )
    fig.add_hline(
        y=95,
        line_dash="dash",
        line_color=COLOR_SERIOUS,
        annotation_text="95 % acceptance threshold",
    )
    save_fig(fig, "08_5_dwd_temporal_coverage")
    fig.show()

# %% [markdown]
# > **Observation.** Spatial coverage meets the criterion: 2,072,398 / 2,092,401
# > accidents (99.0 %) have a DWD station within 30 km. Temporal completeness
# > is more uneven — temperature readings are above 95 % only for 2016–2018;
# > all years 2019–2024 fall to 91–93 %, below the 95 % threshold. Wind speed
# > (50.8 % missing) and visibility (54.4 % missing) have substantial gaps and
# > are flagged as risks in §11. Station distance (0.0 % missing), temperature
# > (7.0 %), and precipitation (9.9 %) are acceptable.
# >
# > **Risk logged to §11.** Temporal coverage shortfall and high wind/visibility
# > missingness are documented as known risks for A³.
#
# ---

# %% [markdown]
# ## 8.6 — Weather feature distributions
#
# Univariate distributions of the four DWD variables. Skewness informs the
# transform recommendations in the §10 preprocessing table: right-skewed
# variables (precipitation, visibility) call for log1p; near-symmetric ones
# (temperature, wind speed) do not.

# %%
if df_weather is not None:
    var_specs = [
        ("dwd_temp_air_2m", "Temperature (°C)", False),
        ("dwd_precip_mm", "Precipitation (mm)", True),
        ("dwd_visibility_m", "Visibility (m)", True),
        ("dwd_wind_speed_ms", "Wind speed (m/s)", False),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in var_specs])
    for i, (col, label, log_y) in enumerate(var_specs):
        row, colnum = divmod(i, 2)
        if col not in df_weather.columns:
            continue
        vals = df_weather[col].dropna()
        fig.add_trace(
            go.Histogram(
                x=vals, nbinsx=80, name=label, marker_color=COLOR_PRIMARY, showlegend=False
            ),
            row=row + 1,
            col=colnum + 1,
        )
        if log_y:
            fig.update_yaxes(type="log", row=row + 1, col=colnum + 1)
    fig.update_layout(height=580, title_text="DWD weather variable distributions")
    save_fig(fig, "08_6_dwd_distributions")
    fig.show()

# %% [markdown]
# **Interpretation:** Air temperature is bimodal — a winter peak around
# 5 °C and a summer peak around 20 °C (range approximately −10 to +30 °C),
# reflecting the seasonal cycle; StandardScaler is still appropriate.
# Precipitation is strongly right-skewed (log-scale y-axis), with most
# station-hours near-zero and a sparse extreme-event tail — log1p
# transformation is appropriate. Visibility shows a bell-shaped distribution
# peaking at 50–70 km; rare low-visibility events form a sparse left tail.
# Wind speed is right-skewed with a modal peak around 3 m/s and a tail
# extending to ~15 m/s.

# %%
if df_weather is not None and "dwd_temp_air_2m" in df_weather.columns:
    monthly = con.execute("""
        SELECT UMONAT AS month,
               AVG(dwd_temp_air_2m)  AS mean_temp,
               SUM(dwd_precip_mm)    AS total_precip,
               COUNT(*) AS n
        FROM df_weather
        WHERE dwd_temp_air_2m IS NOT NULL
        GROUP BY UMONAT
        ORDER BY UMONAT
    """).df()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["total_precip"],
            name="Precipitation sum (mm)",
            marker_color="#5a8db8",
            opacity=0.65,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["mean_temp"],
            name="Mean temp (°C)",
            line=dict(color=COLOR_FATAL, width=2),
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Monthly mean temperature and total precipitation (all accident-years)",
        xaxis_title="Month",
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
        height=DEFAULT_FIG_H,
    )
    fig.update_yaxes(title_text="Precipitation sum (mm)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    save_fig(fig, "08_6_dwd_monthly_seasonal")
    fig.show()

# %% [markdown]
# **Interpretation:** The chart shows the expected Northern European seasonal
# pattern: precipitation peaks in summer (convective storms, months 6–8) with
# a modest secondary elevation in winter. Temperature follows an inverted-U
# curve — near 3 °C in January, peaking around 20 °C in July, returning to
# winter levels by December. Winter months have more icy road conditions
# (STRZUSTAND=2), which is a relevant interaction target with temperature in A³.

# %%
if df_weather is not None and "dwd_precip_mm" in df_weather.columns:
    df_weather["_precip_bucket"] = pd.cut(
        df_weather["dwd_precip_mm"].fillna(-1),
        bins=[-2, 0, 5, 20, float("inf")],
        labels=["dry (0 mm)", "light (0–5 mm)", "moderate (5–20 mm)", "heavy (>20 mm)"],
    )
    sev_precip = (
        df_weather.dropna(subset=["UKATGEORIE"])
        .groupby(["_precip_bucket", "UKATGEORIE"], observed=True)
        .size()
        .reset_index(name="n")
    )
    sev_precip["sev_label"] = sev_precip["UKATGEORIE"].map(
        {
            1: "1 — Fatal",
            2: "2 — Serious injury",
            3: "3 — Slight injury",
        }
    )
    fig = px.bar(
        sev_precip,
        x="_precip_bucket",
        y="n",
        color="sev_label",
        barmode="stack",
        color_discrete_map={
            "1 — Fatal": COLOR_FATAL,
            "2 — Serious injury": COLOR_SERIOUS,
            "3 — Slight injury": COLOR_MINOR,
        },
        title="Severity distribution by precipitation bucket",
        labels={
            "_precip_bucket": "Precipitation bucket",
            "n": "accidents",
            "sev_label": "Severity",
        },
        height=DEFAULT_FIG_H,
    )
    save_fig(fig, "08_6_dwd_severity_precip_bucket")
    fig.show()

# %% [markdown]
# > **Observation.** Precipitation is right-skewed as expected — log1p
# > justified. Visibility is bell-shaped (peak 50–70 km) rather than
# > right-skewed; the §10 transform decision should reflect this. Temperature
# > is bimodal (seasonal), not symmetric; wind speed is right-skewed (peak
# > ~3 m/s, tail to 15 m/s), not symmetric. The seasonal chart confirms a
# > typical Northern European annual cycle. The severity-by-precipitation
# > stacked bar shows only two populated buckets (dry / light rain) with
# > near-identical severity shares — no monotone trend visible, consistent
# > with precipitation's low Cramér's V.
#
# ---

# %% [markdown]
# ## 8.7 — Weather × severity: bivariate associations
#
# Association between weather conditions and accident severity, using the same
# Cramér's V and observation-driven framework as §6. These findings feed
# directly into the §10 decision table.

# %%
if df_weather is not None and "_precip_bucket" in df_weather.columns:
    fig = px.box(
        df_weather.dropna(subset=["UKATGEORIE", "_precip_bucket"]),
        x="_precip_bucket",
        y="UKATGEORIE",
        color="_precip_bucket",
        title="Severity class (UKATGEORIE) by precipitation bucket",
        labels={"UKATGEORIE": "Severity (1=fatal, 3=minor)", "_precip_bucket": ""},
        color_discrete_sequence=px.colors.qualitative.Safe,
        height=400,
    )
    fig.update_layout(showlegend=False)
    save_fig(fig, "08_7_dwd_severity_boxplot")
    fig.show()

# %% [markdown]
# **Interpretation:** The chart shows two precipitation buckets — dry (0 mm)
# and light rain (0–5 mm). Median accident severity is identical across both
# and interquartile ranges overlap completely. The effect size of precipitation
# on severity is negligible in isolation, consistent with the low Cramér's V
# (0.008).

# %%
if df_weather is not None:
    weather_cv_rows = []
    for col in ["_precip_bucket", "dwd_temp_air_2m", "dwd_visibility_m", "dwd_wind_speed_ms"]:
        if col not in df_weather.columns:
            continue
        series = df_weather[col].dropna()
        if series.dtype.name not in ("category", "object"):
            series = pd.qcut(series, q=5, labels=False, duplicates="drop")
        valid_idx = series.dropna().index
        cv = cramers_v(
            df_weather.loc[valid_idx, "UKATGEORIE"].dropna().astype(str),
            series.loc[valid_idx].astype(str),
        )
        weather_cv_rows.append({"feature": col, "cramers_v": round(float(cv), 4)})

    weather_cv_df = pd.DataFrame(weather_cv_rows).sort_values("cramers_v", ascending=False)
    weather_cv_df["feature_label"] = (
        weather_cv_df["feature"].map(FEATURE_LABELS).fillna(weather_cv_df["feature"])
    )
    print(weather_cv_df[["feature_label", "cramers_v"]].to_string(index=False))

    fig = px.bar(
        weather_cv_df,
        x="cramers_v",
        y="feature_label",
        orientation="h",
        title="Cramér's V — DWD weather features vs. accident severity",
        labels={"cramers_v": "Cramér's V", "feature_label": ""},
        color_discrete_sequence=[COLOR_PRIMARY],
        text="cramers_v",
        height=340,
    )
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
    save_fig(fig, "08_7_dwd_cramers_v")
    fig.show()

# %% [markdown]
# **Interpretation:** All four weather features show a weak positive association
# with accident severity. Wind speed has the highest Cramér's V (0.018),
# followed by temperature (0.015); precipitation and visibility are tied at
# 0.008. Effect sizes are modest across the board — weather features complement
# primary predictors (UART, UTYP1) rather than replacing them, and their value
# likely lies in interaction terms rather than standalone signal.

# %%
if df_weather is not None and "dwd_precip_mm" in df_weather.columns:
    monthly_ts = con.execute("""
        SELECT UJAHR AS year, UMONAT AS month,
               COUNT(*) AS n_total,
               SUM(CASE WHEN UKATGEORIE = 1 THEN 1 ELSE 0 END) AS n_fatal,
               100.0 * SUM(CASE WHEN UKATGEORIE = 1 THEN 1 ELSE 0 END)
                   / NULLIF(COUNT(*), 0) AS pct_fatal,
               AVG(dwd_precip_mm) AS mean_precip
        FROM df_weather
        WHERE dwd_precip_mm IS NOT NULL
        GROUP BY UJAHR, UMONAT
        ORDER BY UJAHR, UMONAT
    """).df()
    monthly_ts["period"] = pd.to_datetime(
        monthly_ts["year"].astype(str) + "-" + monthly_ts["month"].astype(str).str.zfill(2) + "-01"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly_ts["period"],
            y=monthly_ts["mean_precip"],
            name="Mean precipitation (mm)",
            marker_color="#5a8db8",
            opacity=0.5,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_ts["period"],
            y=monthly_ts["pct_fatal"],
            name="% fatal",
            line=dict(color=COLOR_FATAL, width=1.5),
            mode="lines",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Monthly mean precipitation vs. fatal accident rate (2016–2024)",
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
        height=DEFAULT_FIG_H,
    )
    fig.update_yaxes(title_text="Mean precipitation (mm)", secondary_y=False)
    fig.update_yaxes(title_text="% fatal accidents", secondary_y=True)
    save_fig(fig, "08_7_dwd_monthly_fatality_precip")
    fig.show()

# %% [markdown]
# > **Observation.** Cramér's V values: Windgeschwindigkeit = 0.018,
# > Lufttemperatur = 0.015, Sichtweite = 0.008, Niederschlagsmenge = 0.008.
# > All four fall below the 0.05 detectable-association threshold — none show
# > a meaningful standalone association with UKATGEORIE. The monthly time series
# > shows no consistent co-movement between precipitation and fatal-accident rate.
# >
# > **Decision.** No feature meets the 0.05 criterion individually. Following
# > the stated rationale — weather signal lies partly in interaction effects
# > with time and location features — all four DWD variables are **included**
# > in the A³ feature set. Wind/visibility missingness (50–54 %) is logged in
# > §11; A³ must apply median imputation inside the Pipeline.
#
# ---

# %% [markdown]
# ## 8.8 — OSM road-context enrichment
#
# `src/unfallatlas/features/spatial.py` and `src/unfallatlas/data/osm.py` add
# road-context features aggregated per H3-8 cell (~0.7 km² hexagons):
# dominant road class, mean/max speed limit, road density, and a
# junction/complexity proxy (distinct-way count). Fetched once from
# OpenStreetMap per Bundesland (16 states), cached to `data/raw/osm/`, then
# joined onto every accident by its H3 cell.
#
# **Known limitation:** OSM reflects the present-day road network; accidents
# span 2016–2024 and some roads' classification/speed limits will have
# changed since. This is an accepted approximation (same category as the DWD
# weather join's day-of-month averaging, §8.5) — not solvable without a paid
# historical-OSM-snapshot service, out of scope here.

# %%
import logging  # noqa: E402
import sys  # noqa: E402

from unfallatlas.data.osm import GERMAN_STATES, build_spatial_features  # noqa: E402
from unfallatlas.features.spatial import ROAD_CLASS_RANK  # noqa: E402

# Enables the log.info(...) progress calls already inside download_road_network/
# build_weather_features to actually show up somewhere. Two handlers, for two
# different execution contexts:
#   - StreamHandler(sys.stdout): visible live in a real Jupyter kernel
#     (e.g. VSCode's interactive window), which renders cell stdout directly.
#   - FileHandler(PROGRESS_LOG): confirmed empirically REQUIRED for
#     `jupyter nbconvert --execute` - nbconvert captures each cell's stdout
#     into the notebook's own cell-output JSON, not into nbconvert's own
#     process-level stdout stream, so redirecting nbconvert's stdout to a
#     file (`nbconvert ... > file.log`) never receives it (reproduced with
#     a minimal test notebook). A FileHandler does a direct OS-level write,
#     bypassing stdout/stdout-capture entirely, so it works the same way
#     regardless of which of the two ways this notebook gets executed.
# osmnx's own ox.settings.log_console was tried first and confirmed NOT to
# work in EITHER context: it deliberately writes to sys.__stdout__ ("print
# explicitly to terminal in case Jupyter has captured stdout"), bypassing
# Jupyter's stdout capture entirely, by design - so it never reaches a
# Jupyter cell's rendered output. ox.settings.log_file=True (set in
# download_road_network) routes osmnx's internal progress messages through
# the standard logging module instead, which propagates to both handlers
# below the normal way.
# force=True re-applies this even if something else already called
# basicConfig earlier in the kernel session.
PROGRESS_LOG = BASE_DIR / "reports" / "u_phase_osm_progress.log"
PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(PROGRESS_LOG)],
    force=True,
)

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"

print(
    f"Fetching OSM road networks for {len(GERMAN_STATES)} states (uses per-state cache if present)..."
)
print(f"Progress log (works under nbconvert too): {PROGRESS_LOG}")
df_spatial = build_spatial_features(df_weather, RAW_DIR, INTERIM_DIR, resolution=8)
print(f"Spatially-enriched frame: {len(df_spatial):,} rows, {df_spatial.shape[1]} columns")

osm_cols = [
    "osm_dominant_road_class",
    "osm_maxspeed_mean",
    "osm_maxspeed_max",
    "osm_road_density",
    "osm_way_count",
]
coverage = df_spatial[osm_cols].notna().mean() * 100
print("\nOSM feature coverage (% of accidents with a matched H3 cell):")
print(coverage.round(1))

# %%
fig = px.histogram(
    df_spatial,
    x="osm_dominant_road_class",
    category_orders={"osm_dominant_road_class": list(ROAD_CLASS_RANK.keys())},
    title="Dominant OSM road class by accident (H3-8 cell)",
)
save_fig(fig, "08_8_osm_road_class_distribution")
fig.show()

# %%
fig = px.histogram(
    df_spatial,
    x="osm_maxspeed_mean",
    nbins=40,
    title="Mean OSM speed limit (km/h) in the accident's H3 cell",
)
save_fig(fig, "08_8_osm_maxspeed_distribution")
fig.show()


# %% [markdown]
# ## 9 — Leakage and chronological-split audit
#
# The final audit combines three related safeguards: suspect-feature leakage,
# chronological separation, and identifier overlap. DWD and OSM consistency checks
# then verify that enriched features describe conditions rather than outcomes.

# %% [markdown]
# ### 9.1  Target-leakage probe
#
# `UART` (Unfallart) and `UTYP1` (Unfalltyp) describe *what kind of accident*
# happened. They are recorded after the event and may partially encode the
# severity outcome. We measure the conditional-entropy reduction:
#
# $$\text{reduction} = 1 - \frac{H(\text{UKATGEORIE} \mid X)}{H(\text{UKATGEORIE})}$$
#
# A reduction near 100 % means the feature definitionally encodes the target.


# %%
def entropy(values: pd.Series) -> float:
    counts = Counter(values.dropna())
    total = sum(counts.values())
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def conditional_entropy(y: pd.Series, x: pd.Series) -> float:
    """H(Y | X) in bits."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) == 0:
        return float("nan")
    n = len(pair)
    h = 0.0
    for _, grp in pair.groupby("x", observed=True):
        p_x = len(grp) / n
        h += p_x * entropy(grp["y"])
    return h


probe = con.execute(
    f"SELECT UKATGEORIE, UART, UTYP1, ULICHTVERH, STRZUSTAND, USTUNDE "
    f"FROM '{DATA}' USING SAMPLE 200000 ROWS (reservoir, 42)"
).df()

H_y = entropy(probe["UKATGEORIE"])
print(f"H(UKATGEORIE) = {H_y:.4f} bits  (marginal entropy of the target)\n")
print(f"{'feature':<12s} {'H(Y|X)':>10s} {'reduction':>12s}  verdict")
print("-" * 55)
probe_results = []
for col in ["UART", "UTYP1", "ULICHTVERH", "STRZUSTAND", "USTUNDE"]:
    H_y_given_x = conditional_entropy(probe["UKATGEORIE"], probe[col])
    reduction = 1 - H_y_given_x / H_y
    flag = "→ LEAKAGE RISK" if reduction > 0.5 else "ok"
    print(f"{col:<12s} {H_y_given_x:>10.4f} {reduction:>11.1%}  {flag}")
    probe_results.append(
        {"feature": col, "H_cond": H_y_given_x, "reduction": reduction, "risk": reduction > 0.5}
    )
probe_df = pd.DataFrame(probe_results)

# %%
# Visualise the entropy reduction per feature.
probe_df_sorted = probe_df.sort_values("reduction", ascending=True)
probe_df_sorted["color"] = probe_df_sorted["risk"].map({True: COLOR_FATAL, False: COLOR_PRIMARY})
probe_df_sorted["feature_label"] = (
    probe_df_sorted["feature"].map(FEATURE_LABELS).fillna(probe_df_sorted["feature"])
)
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=probe_df_sorted["reduction"] * 100,
        y=probe_df_sorted["feature_label"],
        orientation="h",
        marker_color=probe_df_sorted["color"].tolist(),
        text=[f"{r:.1%}" for r in probe_df_sorted["reduction"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>entropy reduction = %{x:.2f}%<extra></extra>",
    )
)
fig.add_vline(
    x=50,
    line_dash="dash",
    line_color="grey",
    annotation_text="50 % threshold",
    annotation_position="top",
)
fig.update_layout(
    title="Conditional-entropy reduction of UKATGEORIE given each feature",
    xaxis_title="reduction (%)",
    yaxis_title="",
    height=380,
    margin=dict(l=80, r=80, t=80, b=40),
    showlegend=False,
)
save_fig(fig, "09_leakage_probe_bars")
fig.show()

# %% [markdown]
# > **Observation.** All probed features reduce UKATGEORIE entropy by less
# > than 5 %: Unfallart = 3.4 %, Unfalltyp = 2.2 %, USTUNDE = 0.3 %,
# > Lichtverhältnisse = 0.1 %, Straßenzustand = 0.0 %. None approach the
# > 50 % trigger threshold — no definitional leakage is present.
# >
# > **Decision rule.** A reduction > 50 % triggers a definitional review of
# > the feature — does it encode information not available at the time of the
# > police report? If yes, the feature is excluded in A³. If no (the
# > association is genuine domain signal), the feature is retained but the
# > finding is documented in the §10 decision table. **Verdict here: all
# > features retained; no review triggered.**

# %% [markdown]
# ### 9.2  Chronological split verification

# %%
split = con.execute(f"""
    SELECT
        CASE WHEN UJAHR <= 2022 THEN 'train (2016–2022)'
             WHEN UJAHR  = 2023 THEN 'val   (2023)'
             ELSE               'test  (2024)' END AS split,
        MIN(UJAHR) AS year_min,
        MAX(UJAHR) AS year_max,
        COUNT(*) AS n,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM '{DATA}'
    GROUP BY split
    ORDER BY year_min
""").df()
split

# %%
split_target = con.execute(f"""
    SELECT
        CASE WHEN UJAHR <= 2022 THEN 'train'
             WHEN UJAHR  = 2023 THEN 'val'
             ELSE               'test'  END AS split,
        UKATGEORIE, COUNT(*) AS n
    FROM '{DATA}'
    GROUP BY split, UKATGEORIE
""").df()
pivot = split_target.pivot(index="split", columns="UKATGEORIE", values="n")
pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pct.columns = ["1 Getötet", "2 Schwer", "3 Leicht"]
pct.reindex(["train", "val", "test"]).round(2)

# %% [markdown]
# ### 9.3  No-overlap check
#
# A sanity check: no `OBJECTID` may appear in more than one split.

# %%
overlap = con.execute(f"""
    WITH labelled AS (
        SELECT OBJECTID,
               CASE WHEN UJAHR <= 2022 THEN 'train'
                    WHEN UJAHR  = 2023 THEN 'val'
                    ELSE               'test'  END AS split
        FROM '{DATA}'
    )
    SELECT OBJECTID, COUNT(DISTINCT split) AS n_splits
    FROM labelled
    GROUP BY OBJECTID
    HAVING COUNT(DISTINCT split) > 1
""").df()
print("OBJECTIDs appearing in more than one split:", len(overlap))

# %% [markdown]
# > **Verdict.** The chronological split is well-formed, class distribution is
# > stable across splits, and no OBJECTID overlap exists. Temporal leakage is
# > structurally prevented.
#
# ---

# %% [markdown]
# ### 9.4  DWD temporal leakage probe
#
# DWD weather features represent meteorological conditions at the hour of the
# accident. The join key is (station_id, year, month, hour-of-day) — all DWD
# observations are historical records that precede or coincide with the accident
# in real time. No future information enters by construction.
#
# The probe below documents the rural-proxy character of `dwd_station_dist_km`.

# %%
if df_weather is not None and "dwd_station_dist_km" in df_weather.columns:
    print("Join key: (UJAHR, UMONAT, USTUNDE) × DWD (year, month, hour-of-day)")
    print("All DWD values are historical observations — no future data can enter.\n")

    n_rural = int((df_weather["dwd_station_dist_km"] > 30).sum())
    n_covered = int((df_weather["dwd_station_dist_km"] <= 30).sum())
    pct_fatal_rural = (
        100
        * (df_weather[df_weather["dwd_station_dist_km"] > 30]["UKATGEORIE"].eq(1).sum())
        / max(n_rural, 1)
    )
    pct_fatal_covered = (
        100
        * (df_weather[df_weather["dwd_station_dist_km"] <= 30]["UKATGEORIE"].eq(1).sum())
        / max(n_covered, 1)
    )
    print(f"  Rural (station > 30 km)   — n = {n_rural:>9,}  fatal rate = {pct_fatal_rural:.2f}%")
    print(
        f"  Covered (station ≤ 30 km) — n = {n_covered:>9,}  fatal rate = {pct_fatal_covered:.2f}%"
    )
    if pct_fatal_rural > pct_fatal_covered * 1.2:
        print("\n  → Rural fatal rate >20 % higher: dwd_station_dist_km is a rural proxy — retain.")
    else:
        print("\n  → No strong rural/urban fatal-rate gap: retain as quality indicator.")
else:
    print("DWD data not loaded — skipping §9.4 probe.")

# %% [markdown]
# > **Verdict.** No temporal leakage from DWD features: the join is on
# > historical hour-aligned observations. `dwd_station_dist_km` encodes rural
# > character, not future information — it is retained as a feature, not
# > removed as a leakage vector.
#
# ---

# %% [markdown]
# ### 9.5 — OSM feature consistency probe
#
# This check mirrors the conditional-entropy method in section 9.1. OSM describes
# the road rather than the crash outcome, so an unexpectedly large entropy
# reduction would indicate a data artefact requiring investigation.

# %%
if "osm_dominant_road_class" in df_spatial.columns:
    baseline_entropy = entropy(df_spatial["UKATGEORIE"])
    cond_entropy = conditional_entropy(
        df_spatial["UKATGEORIE"], df_spatial["osm_dominant_road_class"]
    )
    reduction_pct = 100 * (baseline_entropy - cond_entropy) / baseline_entropy
    print(f"Baseline entropy: {baseline_entropy:.4f}")
    print(f"Conditional entropy given osm_dominant_road_class: {cond_entropy:.4f}")
    print(f"Reduction: {reduction_pct:.1f}%")
    if reduction_pct > 50:
        print("\n  → WARNING: reduction exceeds the 50% trigger - investigate before including.")
    else:
        print("\n  → Below the 50% trigger - retain as a feature.")
else:
    print("OSM data not loaded — skipping §9.5 probe.")

# %% [markdown]
# ## 10 — Target viability and preprocessing decisions
#
# ### Why the operational target becomes binary KSI
#
# The three-class audit produces a coherent feasibility warning:
#
# - the fatal class is only about 1% of observations, although all three shares
#   remain stable across years;
# - the strongest individual target association is only about Cramér's V = 0.13,
#   and lighting, road condition, and weather are much weaker;
# - the public dataset omits impact speed, occupant age, seat-belt use, and
#   vehicle mass—the physical determinants most likely to separate fatal from
#   serious injury.
#
# The original three-class target therefore remains valuable as background and
# negative evidence, but the operational target is **KSI (`UKATGEORIE ≤ 2`) versus
# slight injury (`UKATGEORIE = 3`)**. This revision changes only the label. All
# feature preparation and leakage controls below are target-independent.
#
# ### Preprocessing contract
#
# The following table specifies, per column, what A³ must do. U decides; A³
# implements these steps inside a `Pipeline`, so preprocessing statistics are fit
# on training data only.
#
# | Column | Missing strategy | Encoding | Scaling | Notes |
# |:---|:---|:---|:---|:---|
# | `OBJECTID` | n/a | drop before fit | n/a | identifier only; never a feature |
# | `UJAHR` | n/a | drop before fit | n/a | used to define the split, not as a feature |
# | `UMONAT` | drop row if missing | sin/cos cyclic (period 12) | n/a | seasonality observed |
# | `USTUNDE` | drop row if missing | sin/cos cyclic (period 24) | n/a | strong daily structure |
# | `UWOCHENTAG` | drop row if missing | sin/cos cyclic (period 7) | n/a | weekly structure observed |
# | **`UKATGEORIE`** | drop row if missing | derive KSI label after splitting | n/a | source target; preserve three-class values for audit |
# | `UART` | mode | one-hot or target-encoded | n/a | **§9.1 probe result must be acceptable before inclusion** |
# | `UTYP1` | mode | one-hot or target-encoded | n/a | same as `UART` |
# | `ULICHTVERH` | mode | one-hot | n/a | 3 nominal levels |
# | `STRZUSTAND` | mode | one-hot | n/a | 3 nominal levels |
# | `IstRad … IstSonstig` | none observed | pass-through | n/a | binary |
# | `LON`, `LAT` | drop row if outside DE bbox | none for tree models | `StandardScaler` *only if* a distance-based baseline is added | both retained |
# | `UREGBEZ` | mode | target-encoding with smoothing | n/a | moderate cardinality |
# | `UKREIS` | mode | target-encoding with smoothing | n/a | high cardinality (~400 levels) |
# | `UGEMEINDE` | drop column | n/a | n/a | very-high cardinality, no benefit over `UKREIS` |
#
# ### DWD weather features
#
# | Feature | Missing strategy | Recommended transform | Recommended scaling | EDA finding that drives the decision |
# |:---|:---|:---|:---|:---|
# | `dwd_temp_air_2m` | median per (station, month) | none (near-symmetric) | `StandardScaler` | approximately normal (§8.6 histogram); annual seasonality captured by `UMONAT` sin/cos encoding above |
# | `dwd_precip_mm` | zero-fill where station assigned; NaN otherwise | `log1p` | `StandardScaler` | strongly right-skewed (§8.6 histogram); majority of hours have zero or near-zero precipitation |
# | `dwd_visibility_m` | median imputation | `log1p` | `StandardScaler` | strongly right-skewed (§8.6 histogram); low-visibility tail carries the fog/night signal |
# | `dwd_wind_speed_ms` | median imputation | none | `StandardScaler` | approximately symmetric (§8.6 histogram) |
# | `dwd_station_dist_km` | n/a (always computed) | `log` | none | rural-character proxy (§9.4); retain — removing it discards geographic signal not captured by `UKREIS` alone |
#
# > **Note.** All DWD features represent meteorological conditions at the hour of the accident
# > (historical observations). The (year, month, hour-of-day) averaging introduces day-level
# > noise but no future leakage — see §9.4. Actual `fit_transform` calls happen in A³, not here.
#
# ### OSM road-context features
#
# | Feature | Missing strategy | Recommended transform | Recommended scaling | EDA finding that drives the decision |
# |:---|:---|:---|:---|:---|
# | `osm_dominant_road_class` | mode (or a dedicated "unknown" category) | one-hot | n/a | 15 nominal road classes ranked by literature-established severity/speed association; §9.5 entropy-reduction check must clear the 50% trigger before inclusion |
# | `osm_maxspeed_mean` | median imputation | none (already a natural km/h scale) | `StandardScaler` | speed limit is one of the strongest literature-documented predictors of crash severity specifically (not just occurrence) |
# | `osm_maxspeed_max` | median imputation | none | `StandardScaler` | captures the fastest road touching a mixed-road-class cell, complementing the mean |
# | `osm_road_density` | zero-fill (absence of OSM data in a cell most often reflects genuinely low road density, e.g. remote areas, not a data gap) | `log1p` (right-skewed - most cells have few road-vertex points) | `StandardScaler` | proxy for local traffic exposure |
# | `osm_way_count` | zero-fill (same rationale as `osm_road_density`) | `log1p` | `StandardScaler` | junction/complexity proxy — cells with multiple distinct roads are more likely to be intersections |
#
# > **Note.** OSM road-context reflects the *present-day* network; some roads'
# > classification or posted speed limit will have changed since the earliest
# > accidents in this dataset (2016). This is an accepted approximation — see
# > §8.8 — not a defect to fix here; A³ should note it as a limitation when
# > interpreting SHAP importances for these features in Phase C.
#
# ### Imbalance handling
#
# The original three-class imbalance is approximately 1:18:81; the derived
# KSI share is about 16%. A³ still compares balanced class weights, resampling,
# threshold selection, and appropriate model families rather than assuming that
# the binary reformulation removes imbalance.
#
# ### Cross-validation hint
#
# Time-series semantics. Within the 2016 – 2022 training window, A³ should use year-grouped or chronological folds inside the 2016–2022
# training window, never a random fold design that mixes future years into earlier
# validation folds.
#
# ---

# %% [markdown]
# ## 11 — Risks, decisions, and A³ handoff
#
# ### Evidence summary
#
# - **Coverage and quality:** about 2.09M accidents across 2016–2024; identifiers
#   are unique, geographic bounds are explicit, and missingness is concentrated
#   in selected vehicle and weather fields.
# - **Temporal validity:** severity shares are stable within roughly one
#   percentage point. Train 2016–2022, Validation 2023, and Test 2024 have no
#   `OBJECTID` overlap.
# - **Target viability:** the original 1%/18%/81% target is informative for EDA,
#   but standalone associations are weak (maximum target-related Cramér's V about
#   0.13) and key physical severity determinants are absent. KSI versus slight
#   injury is the justified operational target.
# - **Enrichment:** DWD station coverage reaches 99%, but wind and visibility are
#   missing for roughly half the rows. Present-day OSM context adds road class,
#   speed-limit, density, and complexity proxies with a known historical-snapshot
#   limitation.
# - **Leakage:** all conditional-entropy reductions are below 5%; DWD joins use
#   historical observations and OSM describes road context independently of the
#   accident outcome.
#
# ### Decisions carried into A³
#
# 1. Derive KSI from `UKATGEORIE` only after the chronological split, while
#    retaining the original values for audit and comparison.
# 2. Apply every imputation, encoding, scaling, and target-encoding step inside a
#    fitted pipeline; never learn preprocessing statistics from 2023 or 2024.
# 3. Use year-aware validation within the training window and reserve 2024 for one
#    final evaluation.
# 4. Compare imbalance strategies and model families against macro-F1 and
#    recall(KSI), rather than treating accuracy as evidence of success.
# 5. Treat DWD coverage and present-day OSM context as explicit robustness risks
#    when interpreting model importance.
#
# > **A³ handoff.** The data audit now yields one coherent modelling contract:
# > target-independent preprocessing, a chronological evaluation boundary, the
# > original three-class result as feasibility context, and binary KSI as the
# > operational search target.
