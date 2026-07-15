# Technische Review — Unfallatlas Deutschland (QUA³CK, `feature/a3-phase`)

**Gegenstand:** 3-Klassen-Klassifikation der Unfallschwere (`UKATGEORIE` ∈ {1=Getötet, 2=Schwerverletzt, 3=Leichtverletzt}) auf ~2,09 Mio. Unfällen 2016–2024.
**Gate:** macro-F1 ≥ 0.55 **und** Recall(Klasse 1) ≥ 0.50, chronologischer Split (Train 2016–2022 / Val 2023 / Test 2024).
**Status:** Gate verfehlt. Test-2024: macro-F1 = **0.362**, Recall(1) = **0.649** → FAIL (Champion: `lightgbm_balanced`, getunt).

Diese Review basiert vollständig auf dem committeten Code, den Notebooks und den committeten Ergebnis-Artefakten (`a3_model_comparison.csv`, `a3_model_card.json`) sowie den Confusion-Matrizen darin. Alle Zahlen sind nachgerechnet.

---

## 1 — Diagnose in einem Satz

Das Modell verfehlt das Gate nicht wegen eines behebbaren Implementierungsfehlers, sondern weil in der **3-Klassen-Formulierung mit ~1 % Getöteten die geforderten 0.55 macro-F1 und 0.50 Recall(1) auf gegenüberliegenden Enden derselben Precision-Recall-Front liegen** — und weil die verfügbaren Features (umständliche Unfall-Codes, Wetter, grober OSM-Straßenkontext) die eigentlichen physikalischen Determinanten der Schwere (Aufprallgeschwindigkeit, Alter/Anschnallen der Beteiligten, Fahrzeugmasse) **strukturell nicht enthalten**. Das ist ein Bayes-Ceiling, kein Tuning-Problem.

**Kurz:** Das Modell ist sauber gebaut, aber es löst eine Aufgabe, die mit diesen Daten in dieser Zielformulierung nicht lösbar ist. Rettbar ist es nur durch **Reformulierung des Ziels** (binäres KSI-Framing), nicht durch mehr Tuning.

---

## 2 — Die harte Evidenz (was tatsächlich passiert)

### 2.1 Per-Klassen-Zerlegung des Champions (Test-2024)

Aus der Confusion-Matrix in `a3_model_card.json`:

| Klasse | Support | Recall | Precision | **F1** |
|---|---|---|---|---|
| 1 — Getötet | 2 458 (0,9 %) | 0.649 | **0.036** | **0.069** |
| 2 — Schwer | 41 740 (15,5 %) | 0.416 | 0.226 | **0.293** |
| 3 — Leicht | 224 321 (83,5 %) | 0.600 | 0.912 | **0.724** |
| | | | **macro-F1** | **0.362** |

**Das ist der wichtigste Befund der ganzen Review:** Der Recall-Gate (0.50) ist **bereits erfüllt** (0.649). Was das Gate killt, ist macro-F1 — und macro-F1 wird nicht von Recall(1) gedrückt, sondern von **Precision(1) = 3,6 %** und von **F1(2) = 0.29**. Von 43 942 als „getötet" vorhergesagten Unfällen sind nur 1 595 wirklich tödlich. Klasse 1 trägt mit F1 ≈ 0.07 praktisch nichts zum Mittel bei, egal was man tut.

### 2.2 Das Ceiling ist empirisch belegt, nicht vermutet

Über **alle 19 Konfigurationen** in `a3_model_comparison.csv` (Validierung 2023) liegt das Maximum bei:

- **max macro-F1 = 0.424** (`random_forest_balanced`) — aber mit Recall(1) = **0.212**
- **max Recall(1) = 0.641** (`logistic_regression`) — aber macro-F1 = **0.352**

Die Front ist eindeutig: Jede Konfiguration mit hohem macro-F1 kollabiert Recall(1) auf 0.13–0.25; jede mit hohem Recall(1) deckelt macro-F1 bei ~0.35–0.37. **Keine einzige** der 19 Konfigurationen kommt in die Nähe von *beiden* Schwellen gleichzeitig. Die OSM-Integration hat daran ~+0.004 macro-F1 geändert.

### 2.3 Warum 0.55 arithmetisch fast unmöglich ist

Klasse 3 sitzt bei F1 ≈ 0.72 (nahe ihrem Ceiling). Damit macro-F1 = 0.55 wird, müssten Klasse 1 und 2 **im Mittel F1 ≈ 0.46** erreichen. Für Klasse 1 bedeutet F1 = 0.46 bei einer Basisrate von 0,94 % eine Precision von ~0.46 — das entspricht einem **Odds-Lift von ~90×** gegenüber der Basisrate. Die Features müssten also eine Teilpopulation identifizieren, in der Todesfälle ~46 % statt 0,9 % wahrscheinlich sind. Das leisten Lichtverhältnisse, Straßenzustand, Unfallart und OSM-Straßenklasse nachweislich nicht (siehe 2.4).

### 2.4 Die Features tragen das Signal schlicht nicht (aus der eigenen U-Phase)

Die U-Phase hat das selbst quantifiziert und dokumentiert:

- **Cramér's V gegen `UKATGEORIE`:** stärkstes Feature Unfallart = **0.13**, Unfalltyp = **0.11**, Lichtverhältnisse = 0.02, Straßenzustand = 0.01, alle vier DWD-Wetterfeatures **< 0.02**.
- Severity-Shares sind über **alle** Kategorien von Licht und Straßenzustand **nahezu uniform** (≈ 80 % / 18 % / 2 %). Die Verteilung bewegt sich kaum, egal welchen Feature-Wert man betrachtet.

Das heißt: Selbst die „primären Prädiktoren" haben nur schwache marginale Assoziation, und bedingt auf die Features bleibt die Klassenverteilung praktisch die Basisrate. Das ist die Definition einer Feature-limitierten (Bayes-limitierten) Aufgabe.

### 2.5 Kein Leakage, keine Metrik-Fehlrechnung

Zur Absicherung: Der Conditional-Entropy-Leakage-Probe der U-Phase auf `UART`/`UTYP1` fand Reduktionen von 3,4 % / 2,2 % (weit unter der 50 %-Schwelle). Target-Encoding (`UREGBEZ`/`UKREIS`) wird korrekt nur auf Train gefittet und läuft in §7 innerhalb der `cross_validate`-Folds → fold-sicher. Die DWD-Joins tragen per Konstruktion kein temporales Leakage. Das Ergebnis ist also **echt**, nicht künstlich gedeckelt oder aufgebläht. Es gibt hier nichts „zu reparieren", das plötzlich 0.55 freischaltet.

---

## 3 — Beantwortung der Kernfragen (Ursachen-Attribution)

| Kandidat-Ursache | Verdikt | Begründung |
|---|---|---|
| **Klassenimbalance** | Teil-Ursache, aber **behandelt** | Class-Weights sind gesetzt, Recall(1)=0.65. Imbalance erklärt das *Problem der Front*, aber nicht ihre *Lage* — die Front selbst ist zu tief. |
| **Feature-Schwäche** | **Haupt-Ursache** | Cramér's V ≤ 0.13; severity-Shares uniform; keine Demografie/Aufprallgeschwindigkeit. Das setzt das Ceiling. |
| **Label-Rauschen** | **Sekundäre Ursache** | `UKATGEORIE` = „schwerste Unfallfolge" auf **Unfall**-Ebene. Ob ein Unfall von „leicht" auf „schwer" kippt, hängt oft von Zufall (eine Person minimal stärker verletzt) ab → irreduzibler Rauschanteil, v. a. an der 2↔3-Grenze. |
| **Modellwahl** | **Nicht** die Ursache | LightGBM/CatBoost/XGBoost liegen alle in ±0.01 macro-F1. Boosting ist State-of-the-Art für Tabular; wechseln bringt nichts. |
| **Validierungsdesign** | **Korrekt** (Pluspunkt) | Chronologischer Split, GroupKFold nach Jahr, genau eine Test-Berührung. Vorbildlich — und genau deshalb ist das Ergebnis nicht durch Random-Split-Optimismus geschönt. |
| **Leakage-Vermeidung** | **Sauber** | Siehe 2.5. Eher *zu* konservativ als zu lax. |
| **Metrik-Konflikt** | **Real und unterschätzt** | macro-F1 (belohnt alle drei Klassen gleich) und Recall(1)-Gate ziehen aktiv gegeneinander. Der Threshold-Mover optimiert sogar die *falsche* Seite (siehe 4.2). |
| **Performance-Ceiling** | **Bestätigt** | Empirisch 0.42 macro-F1 über 19 Configs; arithmetisch ~90× Odds-Lift für Klasse-1 nötig. Das Gate 0.55 ist in der 3-Klassen-Form außer Reichweite. |

**Attribution (grob):** ~60 % Feature-/Ceiling-limitiert, ~25 % Metrik-/Ziel-Konflikt (0.55 als 3-Klassen-Ziel unrealistisch angesetzt), ~15 % Label-Rauschen. Der Code selbst ist zu < 5 % schuld.

---

## 4 — Warum die OSM-Features fast nichts brachten (+0.004 macro-F1)

Sechs konkrete, sich verstärkende Gründe:

1. **Falsche Granularität.** Alle OSM-Features sind auf H3-Res-8-Zellen (~0,7 km²) aggregiert (`aggregate_roads_to_h3`) und per Zelle auf **jeden** Unfall darin gejoint. Innerhalb einer Zelle ist die Severity-Varianz riesig; ein zell-konstanter Vektor kann sie nicht trennen.
2. **Redundanz/Kollinearität.** `osm_dominant_road_class` und `osm_maxspeed` korrelieren stark mit bereits vorhandenem Signal (`UART`/`UTYP1`, Straßenzustand, Region-Target-Encoding). Die Bäume hatten den Löwenanteil dieser Information schon.
3. **Proxy statt Ursache.** `maxspeed` ist das *zulässige* Tempo, nicht die *Aufprall*geschwindigkeit — die U-Phase nennt letztere selbst „arguably the single strongest physical determinant". OSM kann sie prinzipiell nicht liefern.
4. **Zeitlicher Mismatch.** Present-day-OSM auf Unfälle 2016–2024 (dokumentiert). Klassifikationen/Limits haben sich geändert → zusätzliches Rauschen.
5. **Vertex-gewichtete Dichte.** `osm_road_density` zählt Vertices, nicht Länge/Exposure; ein Weg mit mehr Stützpunkten zählt mehr. Verrauschter Proxy.
6. **Precision-Ceiling ist unberührbar.** Selbst ein perfektes Kontextfeature kann die Precision-Grenze der 1 %-Klasse nicht heben. OSM adressiert den Engpass (Precision Klasse 1, F1 Klasse 2) an keiner Stelle.

Das ist übrigens genau das, was die Cramér's-V-Analyse vorhergesagt hätte — die OSM-Features hätten *vor* dem teuren 16-Bundesländer-Fetch mit dem gleichen bivariaten Assoziations-Check bewertet werden sollen, den die U-Phase für Wetter schon anwendet.

---

## 5 — Methodische Fehler & Fallstricke im Code

Keiner davon „erklärt" den Miss allein, aber alle verzerren die Strategie oder verschenken den einen billigen Hebel, der noch existiert:

1. **Threshold Moving ist architektonisch totgelegt.** `_build_pipeline_for` (A³ §6) wirft `NotImplementedError` für alles außer `{family}_balanced`/`{family}_unweighted`. Damit kann eine gewinnende Threshold-Moving/SMOTE/ADASYN/Ordinal-Strategie **nie** in §7/§8 refittet und aufs Testset gebracht werden. Der laut Literatur-Anker vielversprechendste Hebel ist als Sackgasse implementiert.

2. **Der Threshold-Mover optimiert die falsche Zielgröße.** `find_best_threshold_for_class` sweept den Schwellwert, um **macro-F1** zu maximieren. Auf einem ungewichteten Modell degeneriert das zur Fast-Majority-Vorhersage (siehe `lightgbm_threshold_moving`: Recall(3)=0.986, Recall(1)=**0.113**). Das Gate verlangt aber Recall(1)≥0.50 — der Optimierer drückt Recall(1) also aktiv **gegen** das Gate.

3. **Nur die Klasse-1-Schwelle wird bewegt.** Klasse 2 bleibt komplett auf Argmax. Da F1(2)=0.29 der zweite Engpass ist, lässt der Ein-Klassen-Threshold genau den Hebel liegen, der macro-F1 am ehesten heben würde.

4. **SMOTE/ADASYN im transformierten Raum.** Resampling läuft auf dem bereits one-hot-/target-/zyklisch-kodierten Array. Interpolation zwischen One-Hot-Kategorien erzeugt fraktionale Nonsens-Punkte; Ergebnis: Recall(1) kollabiert auf 0.01–0.03. SMOTE hilft Tree-Ensembles auf solchen Tabulardaten praktisch nie — die Resultate bestätigen das lehrbuchmäßig.

5. **Ordinal (Frank-Hall) mit ungewichteten Basis-Lernern.** Jeder binäre Split P(y>1), P(y>2) ist selbst imbalanciert; ungewichtet sagt jeder die Mehrheitsseite → Kollaps auf Klasse 3 (macro-F1 ≈ 0.33). Ordinal wurde also nie fair getestet (keine Class-Weights, kein Tuning). Das ist kein Beweis gegen ordinales Lernen, sondern ein Test-Design-Fehler.

6. **Literatur-Anker vs. Realität.** Der Projektplan projiziert „CatBoost + Threshold → 0.65–0.72 macro-F1". Realität: 0.36. Die zitierten Studien (Santos 2022 ~0.60, Pakgohar 2021 ~0.62, Schlößler 2024 ~0.65) sind mit hoher Wahrscheinlichkeit **nicht vergleichbar**, weil sie typischerweise (a) Personen-/Fahrzeug-Level-Kovariaten inkl. Alter besitzen, (b) ein **binäres** KSI-vs-slight-Framing verwenden, oder (c) — sehr häufig in dieser Literatur — Resampling **vor** dem Split anwenden bzw. auf balanciertem Testset evaluieren (Leakage/optimistischer Bias). Der Anker taugt nicht als realistisches Ziel für *diese* Daten in *dieser* Formulierung.

7. **Gate-Design selbst.** Ein UND-Gate aus einer Makro-Aggregatmetrik (0.55) und einem Minderheits-Recall (0.50) über einer 1 %-Klasse ist ein in sich fast widersprüchliches Ziel. Das gehört im Bericht als methodische Erkenntnis benannt, nicht als Versagen.

---

## 6 — Priorisierte Roadmap

Bewertung je Maßnahme: **Impact** (auf das Gate) · **Aufwand** · **Risiko** · **Best-Practice-Plausibilität** · **Fit** zum Code.

### Kategorie 1 — Kleine Änderungen, hoher Nutzen (zuerst)

| # | Maßnahme | Impact | Aufwand | Risiko | Plausibilität | Fit |
|---|---|---|---|---|---|---|
| 1.1 | **Threshold-Mover auf das Gate-Objektiv umstellen** statt macro-F1 (maximiere macro-F1 **unter** Nebenbedingung Recall(1)≥0.50; oder maximiere gate-gewichteten Score). | mittel | gering | gering | hoch | hoch |
| 1.2 | **Beide Minderheitsschwellen** (Klasse 1 **und** 2) gemeinsam bewegen (2D-Sweep / `predict_proba`-Argmax mit Klassen-Offsets). | mittel–hoch | gering | gering | hoch | hoch |
| 1.3 | **Threshold-Sackgasse in `_build_pipeline_for` schließen**: Threshold-Moving als Post-hoc-Wrapper um das refittete `*_balanced`-Modell (kein Refit nötig, nur Schwellen aus Val übernehmen). | Voraussetzung für 1.1/1.2 | gering | gering | hoch | hoch |
| 1.4 | **Kalibrierung** (`CalibratedClassifierCV`, isotonic, auf Val) vor dem Thresholding — macht Schwellen stabil und übertragbar. | gering–mittel | gering | gering | hoch | hoch |
| 1.5 | **Metrik-Report erweitern**: PR-Kurven pro Klasse, macro-F1-vs-Recall(1)-Frontkurve explizit plotten. Zeigt dem Gutachter die Front — verwandelt den „Miss" in eine belegte Erkenntnis. | (Reporting) | gering | keins | hoch | hoch |

> Realistischer Effekt von Kat. 1 auf die **3-Klassen**-Metrik: macro-F1 evtl. 0.36 → ~0.40–0.43. **Das schließt die Lücke zu 0.55 nicht.** Der Wert liegt in einem sauberen, gate-orientierten Operating Point und im Reporting.

### Kategorie 2 — Mittlerer Aufwand

| # | Maßnahme | Impact | Aufwand | Risiko | Plausibilität | Fit |
|---|---|---|---|---|---|---|
| 2.1 | **Fold-sicheres Resampling via `imblearn.pipeline.Pipeline`** in Optuna-CV (der in der Retrospektive deferrte Punkt). | gering–mittel | mittel | mittel | mittel | mittel |
| 2.2 | **Ordinal fair nachziehen**: Frank-Hall mit **class-weighted** Basis-Lernern + eigenem Tuning. | gering–mittel | mittel | gering | mittel–hoch | mittel |
| 2.3 | **Optuna gezielt auf die Front tunen** (multi-objektiv: `directions=["maximize","maximize"]` für macro-F1 & Recall(1), Pareto-Front). | gering | mittel | gering | hoch | hoch |
| 2.4 | **Cost-sensitive Boosting** mit realen Kostenmatrizen statt „balanced" (z. B. FN(getötet) teurer). | gering–mittel | mittel | mittel | hoch | hoch |

> Die Retrospektive vermutet bereits (korrekt), dass mehr Optuna-Budget wenig bringt — die Front ist ein Plateau, kein Suchproblem. Kat. 2 ist v. a. für Vollständigkeit/Portfolio wertvoll, nicht als Gate-Retter. **Ehrliche Erwartung: kein Config in Kat. 1+2 erreicht 0.55 in der 3-Klassen-Form.**

### Kategorie 3 — Hoher Aufwand (neue Features)

| # | Maßnahme | Impact | Aufwand | Risiko | Plausibilität | Fit |
|---|---|---|---|---|---|---|
| 3.1 | **Personen-/Fahrzeug-Level-Kovariaten** (Alter, Anschnallen, Fahrzeugtyp/-masse, #Beteiligte). **Der eigentliche Hebel** — aber **nicht im Unfallatlas enthalten**. | **hoch** (wenn beschaffbar) | sehr hoch / evtl. unmöglich | hoch | sehr hoch | niedrig |
| 3.2 | **Aufprall-/tatsächliche Geschwindigkeit** als Proxy verbessern (Tempolimit × Tageszeit × Straßenklasse-Interaktion, Steigungen, Kurvenradien aus OSM-Geometrie). | gering–mittel | hoch | mittel | mittel | mittel |
| 3.3 | **Bessere Exposure-Daten** (Verkehrsdichte/DTV, ÖPNV-Nähe, Bevölkerungsdichte) — verbessert v. a. *Häufigkeits*-, weniger *Schwere*-Vorhersage. | gering | hoch | mittel | mittel | niedrig |
| 3.4 | **Wetter zum exakten Zeitpunkt** statt Tages-/Monatsaggregat (setzt `UTAG` voraus, der in `accidents.parquet` fehlt). | sehr gering | hoch | gering | niedrig | niedrig |

> Realistätscheck: Der einzige Feature-Block mit echtem Ceiling-Hebel ist 3.1 — und der ist im offenen Datensatz **nicht verfügbar** (die Destatis-Mikrodaten mit Personenbezug sind zugangsbeschränkt). 3.2–3.4 sind Feinschliff an einem Proxy, der bereits < 0.02 Cramér's V zeigt. **Kein realistischer Weg zu 0.55 über Features im Rahmen dieses Projekts.**

### Kategorie 4 — Strategische Alternativen (hier liegt die Lösung)

| # | Maßnahme | Impact | Aufwand | Risiko | Plausibilität | Fit |
|---|---|---|---|---|---|---|
| **4.1** | **Binäres KSI-Framing: leicht (3) vs. schwer|getötet ({1,2})** | **hoch** | gering–mittel | gering | **sehr hoch** | hoch |
| 4.2 | **Hierarchische Pipeline**: Stufe 1 KSI-vs-slight, Stufe 2 innerhalb KSI fatal-vs-schwer. | mittel | mittel | mittel | hoch | mittel |
| 4.3 | **3-Klassen-Ziel als „nicht realistisch lösbar" dokumentieren** und Gate anpassen. | (Reporting) | gering | keins | hoch | hoch |

**4.1 ist die zentrale Empfehlung — und sie ist evidenzbasiert, nicht spekulativ:**

Ich habe die **vorhandenen** Champion-Vorhersagen (ohne jedes Neutraining) in KSI={1,2} vs. slight={3} umgelabelt:

| | Recall | Precision | F1 |
|---|---|---|---|
| KSI (schwer/getötet) | 0.708 | 0.259 | 0.379 |
| slight (leicht) | 0.600 | 0.912 | 0.724 |
| | | **binär macro-F1** | **0.552** |

Schon das **naive Umlabeln** eines für 3 Klassen optimierten Modells erreicht **macro-F1 = 0.552** — auf der Schwelle. Ein *direkt für binäres KSI trainiertes und geschwelltes* Modell wird das deutlich übertreffen (realistisch **0.58–0.65**), weil es die Entscheidungsgrenze direkt optimieren und den Operating Point frei wählen kann. KSI-vs-slight ist außerdem der **Standard** der Verkehrssicherheits-ML-Literatur (u. a. weil genau dieses Ceiling seit Jahren bekannt ist).

**Die einzige Konsequenz:** Ein separater `Recall(Klasse getötet) ≥ 0.50` ist im binären Framing nicht mehr definiert. Optionen: (i) Gate auf `Recall(KSI) ≥ 0.50 & macro-F1 ≥ 0.55` umstellen — beides erfüllbar; (ii) hierarchisch (4.2), um eine Fatal-Recall-Größe zu behalten — aber Stufe 2 (fatal ≈ 5 % *innerhalb* KSI) ist erneut brutal precision-limitiert (aktuell: von 43 942 „getötet"-Vorhersagen sind 3,6 % korrekt).

---

## 7 — Konkrete code-nahe Empfehlungen

**A) Threshold-Sackgasse öffnen (Kat. 1.3) — Post-hoc-Wrapper statt Refit-Pfad**

In `models/imbalance.py` das Ein-Klassen-Objektiv durch ein gate-bewusstes, 2-Klassen-Objektiv ersetzen:

```python
def find_gate_optimal_offsets(y_true, y_proba, classes,
                              recall_gate_class=1, recall_gate=0.50):
    """Sweep additive log-prob offsets for the two minority classes;
    maximise macro-F1 SUBJECT TO recall(gate_class) >= recall_gate.
    Falls back to max macro-F1 if the constraint is infeasible."""
    classes = list(classes); best = (None, -1.0)
    for o1 in np.linspace(0, 3.0, 13):        # boost class 1
        for o2 in np.linspace(0, 2.0, 11):    # boost class 2
            logit = np.log(np.clip(y_proba, 1e-9, 1)).copy()
            logit[:, classes.index(1)] += o1
            logit[:, classes.index(2)] += o2
            y_pred = np.array(classes)[logit.argmax(1)]
            r1 = recall_score(y_true, y_pred, labels=[recall_gate_class], average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")
            if r1 >= recall_gate and f1 > best[1]:
                best = ((o1, o2), f1)
    return best
```
Offsets auf **Val 2023** bestimmen, unverändert auf **Test 2024** anwenden. Kein Eingriff in `_build_pipeline_for` nötig — läuft komplett post-hoc auf `predict_proba` des refitteten `*_balanced`-Modells.

**B) Binäres KSI-Target (Kat. 4.1) — minimaler Eingriff**

`preprocessing.py::split_features_target` bekommt eine Variante:

```python
def split_features_target_binary(df):
    y = (df[TARGET_COLUMN].astype(int) <= 2).astype(int)  # 1 = KSI, 0 = slight
    X = df.drop(columns=[TARGET_COLUMN, SPLIT_YEAR_COLUMN])
    return X, y
```
Danach identische Pipeline, aber `LGBMClassifier(objective="binary", is_unbalance=True)` bzw. `scale_pos_weight`, und Threshold auf Val optimieren. Evaluate-Gate auf `macro_f1 >= 0.55 & recall(KSI) >= 0.50` umstellen. **Erwartung: Gate erfüllbar.**

**C) Ordinal fair testen (Kat. 2.2)** — in `ordinal.py` den Basis-Lerner class-weighted übergeben (`class_weight="balanced"` bzw. `sample_weight` je binärem Subproblem), sonst ist der Vergleich wertlos.

**D) Reporting (Kat. 1.5)** — die macro-F1-vs-Recall(1)-Front aus `a3_model_comparison.csv` als Scatter plotten und die Gate-Zielmarke (0.55/0.50) einzeichnen. Das ist die überzeugendste Abbildung für die C-Phase: sie *zeigt*, dass das Gate außerhalb der erreichbaren Front liegt.

---

## 8 — Methodische Fehler (Kurzliste zum Abhaken)

1. Threshold-Moving optimiert macro-F1 statt des Gates → drückt Recall(1) gegen 0.
2. Threshold-Moving/SMOTE/ADASYN/Ordinal können nie ins Testset (NotImplementedError-Sackgasse).
3. SMOTE/ADASYN auf one-hot/target-kodiertem Raum (fraktionale Kategorien).
4. Ordinal ungewichtet → nie fair getestet.
5. Literatur-Anker (0.65–0.72) nicht auf Vergleichbarkeit geprüft (Framing/Daten/Leakage der Quellen).
6. OSM-Features ohne vorherigen bivariaten Assoziations-Check gebaut (der teure Fetch war vermeidbar riskant).
7. Gate-Design (Makro-Metrik UND Minderheits-Recall) intern fast widersprüchlich — als Erkenntnis, nicht als Fehler zu framen.

---

## 9 — Ist das Modell rettbar? Lohnt weiteres Tuning?

**In der 3-Klassen-Form: nein, nicht auf 0.55.** Das Ceiling ist empirisch (0.42 über 19 Configs) und arithmetisch (~90× Odds-Lift für Klasse 1) belegt. Weiteres Optuna-/Sampling-/Feature-Tuning innerhalb dieser Formulierung ist **verlorene Zeit** — die eigene Retrospektive sieht das Plateau bereits, und die Cramér's-V-Analyse erklärt, warum. Das ist keine Schwäche der Arbeit, sondern eine korrekte, gut dokumentierte Grenze der offenen Daten.

**Als Projekt: ja, klar rettbar — durch Reframing.** Der binäre KSI-Pfad (4.1) erreicht das Ziel schon beim Umlabeln vorhandener Predictions (macro-F1 = 0.552) und wird purpose-built darüber liegen. Aufwand: gering–mittel, Risiko: gering, Best-Practice: Standard der Domäne.

**Empfohlene Reihenfolge (rein wirtschaftlich):**
1. **Binäres KSI-Modell** bauen + Gate anpassen (4.1/4.3) → erfüllt das Ziel.
2. **3-Klassen-Version als „ehrliche Grenze" behalten**, Front-Plot + Ceiling-Argument in die C-Phase (das ist *wissenschaftlich* der stärkere Beitrag als ein erzwungenes 0.55).
3. Optional: hierarchische Pipeline (4.2) für eine erhaltene Fatal-Recall-Kennzahl — mit ehrlicher Diskussion, dass Stufe 2 precision-limitiert bleibt.
4. **Nicht** weiter in 3-Klassen-Tuning, SMOTE-Varianten oder zusätzliche OSM-Ableitungen investieren.

**Note-technisch** (gemäß eurem Projektplan-Bewertungsraster): Ein sauber begründetes „das 3-Klassen-Gate ist mit offenen Daten nicht erreichbar, hier ist die Evidenz und hier ist die tragfähige binäre Alternative" ist ein **1,0-Argument** — es demonstriert genau die kritische, limitationsbewusste Reflexion, die der Plan selbst als das seltene Unterscheidungsmerkmal nennt. Ein auf Biegen und Brechen erzwungenes 0.55 wäre schwächer und methodisch fragwürdiger.