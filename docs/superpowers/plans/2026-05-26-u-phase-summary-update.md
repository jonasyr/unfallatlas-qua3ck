# U-Phase §11 Summary Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `notebooks/02_U_Phase.ipynb` cell 102 (§11 Summary) to reflect the actual results found during the U-phase run, replacing three stale/wrong claims.

**Architecture:** Single markdown cell edit to the `.ipynb` JSON. No code cells touched.

**Tech Stack:** Python JSON editing of `.ipynb`, git commit.

---

## What changed vs. what §11 currently says

| Section | Current (stale) | Actual result |
|---------|----------------|---------------|
| Top risk 1 | "If the §9.1 probe reports reduction > 50 %, features must be excluded" | Probe ran: Unfallart = 3.4 %, Unfalltyp = 2.2 % — far below 50 %; both retained |
| Top risk 4 | "Up to ~5 % of accidents lack a station within 30 km" | Actual: 99.0 % within 30 km — only ~1 % lack a station |
| Top risk 4 | "If §8.5 shows coverage < 95 %, revisit max_km" | Spatial coverage = 99 % (resolved); the actual DWD risks are temporal completeness < 95 % for 2019–2024 and wind/visibility 50–54 % missingness |
| Checklist | "right-skew confirmed for precip/visibility" | Visibility is bell-shaped (peak 50–70 km), not right-skewed |

Everything else in §11 remains accurate.

---

## Files

- Modify: `notebooks/02_U_Phase.ipynb`, cell index 102

---

## Task 1 — Update §11 Summary: risks and checklist

**Files:** Modify `notebooks/02_U_Phase.ipynb`, cell index 102

- [ ] **Step 1: Edit cell 102 source**

Replace the full source with:

```python
new_source = [
    "## 11 — Summary\n",
    "\n",
    "### Dataset characterisation\n",
    "\n",
    "- **Volume:** ~2.09 M rows · 21 columns · 9 vintages 2016 – 2024.\n",
    "- **Quality:** no duplicate OBJECTIDs; row-duplicate count negligible;\n",
    "  geographic outliers handled by explicit bounding box; missingness\n",
    "  concentrated in `IstGkfz` and the four DWD weather columns.\n",
    "- **Target:** class imbalance ≈ 1 % / 18 % / 81 %, stable across years and\n",
    "  splits.\n",
    "- **Splits:** chronological, train 2016 – 2022 / val 2023 / test 2024;\n",
    "  no OBJECTID overlap; class proportions stable across splits.\n",
    "- **Patterns:** bimodal hourly distribution (morning 7–9 h, afternoon 15–17 h\n",
    "  peaks); severity inverts the count signal (more severe at night and\n",
    "  weekends); Thüringen and Sachsen-Anhalt carry the highest fatal-accident\n",
    "  shares; urban / rural split is real but not clean.\n",
    "- **Weather enrichment:** DWD CDC hourly weather joined by nearest station\n",
    "  within 30 km; four variables (temperature, precipitation, visibility,\n",
    "  wind speed) added; 99 % spatial coverage; temporal completeness < 95 %\n",
    "  for 2019 – 2024 (91 – 93 %); wind and visibility have 50–54 % missing\n",
    "  values. Cramér's V vs. UKATGEORIE: max 0.018 (wind speed).\n",
    "- **Leakage:** conditional-entropy probe on `UART` / `UTYP1` found\n",
    "  reductions of 3.4 % and 2.2 % — both far below the 50 % trigger; all\n",
    "  features retained. DWD features carry no temporal leakage by join-key\n",
    "  construction (§9.4).\n",
    "\n",
    "### Top-4 risks for A³\n",
    "\n",
    "1. **Imbalance collapse on macro-F1.** Without class weights or sampling,\n",
    "   tree models default to majority-class prediction on minority instances;\n",
    "   recall on class 1 will fall below the 0.50 acceptance threshold.\n",
    "2. **DWD wind / visibility missingness (50–54 %).** These two features have\n",
    "   limited effective sample size. A³ must apply median imputation inside\n",
    "   the Pipeline; any model that relies heavily on these features will have\n",
    "   degraded coverage on roughly half the dataset.\n",
    "3. **DWD temporal completeness shortfall.** Temperature (and by extension\n",
    "   the other DWD readings) falls below 95 % valid readings for every year\n",
    "   from 2019 onward (91 – 93 %). The remaining ~7–9 % of accident-hours\n",
    "   have no matched DWD record and require imputation.\n",
    "4. **Stationarity assumption between 2016 – 2022 and 2024.** COVID-19\n",
    "   produced a structural year (2020). If A³ trains naively, the model\n",
    "   learns the COVID-year distribution as if it were normal; consider a\n",
    "   year-weight or drop 2020 from training and document the choice.\n",
    "\n",
    "### U-phase acceptance checklist\n",
    "\n",
    "```text\n",
    "[ ] Provenance block at top — versions, hash, git commit, seed\n",
    "[ ] Schema printed and annotated with semantic types\n",
    "[ ] Cardinality + missingness table rendered\n",
    "[ ] Missingness map on a sample rendered\n",
    "[ ] Sentinel-value scan executed\n",
    "[ ] Duplicate detection (OBJECTID + exact rows)\n",
    "[ ] Range / domain bound checks on coordinates and ordinals\n",
    "[ ] Consistency rule check (at least one transport mode set)\n",
    "[ ] Target distribution + imbalance ratio computed\n",
    "[ ] Target stability across years verified\n",
    "[ ] Univariate countplots / histograms for all relevant features\n",
    "[ ] Cramér's V matrix rendered\n",
    "[ ] Conditional severity plots for two key features\n",
    "[ ] Hourly profile + weekday × hour heatmaps\n",
    "[ ] Geographic density map + Bundesland aggregate\n",
    "[ ] DWD station coverage ≥ 95 % of accidents within 30 km\n",
    "[ ] DWD temporal completeness per year (§8.5) — note shortfall if < 95 %\n",
    "[ ] DWD univariate distributions rendered — right-skew confirmed for precip;\n",
    "    visibility bell-shaped; temperature bimodal; wind right-skewed\n",
    "[ ] DWD monthly seasonal chart rendered\n",
    "[ ] DWD severity-by-precipitation stacked bar rendered\n",
    "[ ] DWD Cramér's V computed for all 4 variables vs. UKATGEORIE\n",
    "[ ] DWD monthly fatality-vs-precipitation time-series rendered\n",
    "[ ] DWD §10 rows filled (missing strategy, recommended transform, recommended scaling)\n",
    "[ ] DWD temporal leakage probe executed (§9.4) — no future data by construction\n",
    "[ ] Conditional-entropy leakage probe executed for UART, UTYP1\n",
    "[ ] Chronological split sizes verified\n",
    "[ ] Class stability across splits verified\n",
    "[ ] No OBJECTID overlap between splits\n",
    "[ ] §10 preprocessing decision table filled per column (Unfallatlas + DWD)\n",
    "[ ] Top-4 risks for A³ written\n",
    "[ ] All plots exported to reports/figures/u_phase/\n",
    "[ ] Notebook runs end-to-end without manual intervention\n",
    "```\n",
    "\n",
    "> **Transition.** The dataset is audited, the leakage probes are run, and\n",
    "> the preprocessing contract is written. Proceed to `03_A3_Phase.ipynb` to\n",
    "> implement the decisions above and train the first baseline models."
]
```

- [ ] **Step 2: Verify**

Spot-check the rendered cell: confirm risk 1 now leads with imbalance (not leakage), risk 4 is stationarity, and the checklist DWD distribution item reads "visibility bell-shaped".

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_U_Phase.ipynb
git commit -m "fix(notebooks): update §11 summary — actual DWD/leakage results replace stale conditionals"
```

---

## Self-review

**Spec coverage:**
- Top risk 1 (leakage conditional) ✓ replaced with imbalance as primary risk
- Top risk 4 (~5 % spatial gap) ✓ replaced with wind/visibility missingness + temporal shortfall
- Checklist DWD distributions ✓ visibility corrected from right-skewed to bell-shaped
- Everything else in §11 unchanged ✓

**Placeholder scan:** No TBDs. All new text cites actual numbers from observed charts.

**Type consistency:** N/A — markdown only.
