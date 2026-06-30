---
title: Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung
description: ""
date: 12-05-2026
time: 12:19
reference: Data Analytics und Big Data
index: ""
subindex: ""
status:
  - begin
---

# Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung

>- **Reference Link:** [[Data Analytics und Big Data]]

---
>[!summary]
> Diese Einheit vertieft die **U-Phase** des QUA³CK-Modells: **Understanding the Data**.
> Im Mittelpunkt steht die Frage, wie Rohdaten vor der eigentlichen Modellierung systematisch untersucht, bereinigt, visualisiert und vorbereitet werden. Genau hier entscheidet sich oft, ob ein Machine-Learning-Projekt später brauchbare Ergebnisse liefert oder nur hübsch scheitert, wie so viele digitale Hoffnungsprojekte der Menschheit.

---

### 1. Warum ist „Understanding the Data“ so wichtig?

Die Datenphase ist das Fundament eines Machine-Learning-Projekts.

Bevor ein Algorithmus trainiert wird, muss klar sein:

- welche Daten überhaupt vorliegen

- aus welchen Quellen sie stammen

- welche Struktur sie besitzen

- welche Qualität sie haben

- welche Werte fehlen oder auffällig sind

- welche Variablen miteinander zusammenhängen

- welche Vorverarbeitung notwendig ist

>[!important]

> Ein Modell kann nur so gut sein wie die Daten, auf denen es trainiert wird.

>

> Schlechte Daten führen nicht zu „kreativen“ Modellen, sondern zu schlechten Modellen mit mathematischem Selbstbewusstsein.

In der Praxis entfällt ein großer Teil der Arbeit in Data-Science-Projekten auf Datenverständnis und Datenvorbereitung. Je nach Definition und Studie werden etwa **45 % bis 80 %** des Arbeitsaufwands für Datenaufbereitung, Bereinigung, Exploration und Vorbereitung aufgewendet.

Das bedeutet:

```text

Daten verstehen → Daten bereinigen → Daten vorbereiten → sinnvoll modellieren

```

Ohne diese Phase sind spätere Schritte wie Algorithmusauswahl, Feature Engineering, Training und Evaluation kaum zuverlässig möglich.

---

### 2. Einordnung in das QUA³CK-Modell

Die Einheit gehört zur zweiten Phase des QUA³CK-Prozessmodells.

| Phase | Bedeutung | Rolle in dieser Einheit |

| ---- | ---- | ---- |

| **Q** | Question | Forschungsfrage und Ziel klären |

| **U** | Understanding the Data | Daten analysieren, visualisieren und vorbereiten |

| **A³** | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | Modelle auswählen, Features anpassen, Hyperparameter optimieren |

| **C** | Conclude & Compare | Modelle bewerten und vergleichen |

| **K** | Knowledge Transfer | Ergebnisse dokumentieren und nutzbar machen |

>[!note]

> Einheit 2 behandelt vor allem die **U-Phase**.

>

> Sie bildet die Brücke zwischen der Problemdefinition aus Einheit 1 und der späteren Modellierung.

---

### 3. Lernziele der Einheit

Nach dieser Einheit solltest du:

- erklären können, warum Datenverständnis für Machine Learning zentral ist

- verschiedene **Datenquellen** und **Datentypen** unterscheiden können

- Ziele und Methoden der **Exploratory Data Analysis (EDA)** kennen

- passende Visualisierungen für unterschiedliche Datenarten auswählen können

- fehlende Werte erkennen, analysieren und sinnvoll behandeln können

- einfache und fortgeschrittene Imputationsmethoden unterscheiden können

- Skalierung, Normalisierung und Standardisierung erklären können

- typische Fehler wie **Data Leakage** vermeiden können

- eine Datenanalyse in einem Jupyter Notebook sauber dokumentieren können

---

### 4. Datenquellen im Machine Learning

Daten können aus sehr unterschiedlichen Quellen stammen. Für ein ML-Projekt ist wichtig, die Herkunft und Eigenschaften der Daten zu dokumentieren.

| Datenquelle | Beschreibung | Beispiele |

| ---- | ---- | ---- |

| **Relationale Datenbanken** | Strukturierte Daten in Tabellenform | PostgreSQL, MySQL, Oracle |

| **APIs & Web-Scraping** | Daten aus Schnittstellen oder Webseiten | REST APIs, Twitter/X API, BeautifulSoup |

| **Öffentliche Datensätze** | Frei verfügbare Datensammlungen | UCI Repository, Kaggle, Google Dataset Search |

| **Unternehmensinterne Daten** | Daten aus betrieblichen Systemen | CRM, ERP, Server-Logs, proprietäre Datenbanken |

| **Unstrukturierte Daten** | Daten ohne feste Tabellenstruktur | Texte, Bilder, Videos, Audio, Social-Media-Beiträge |

>[!tip]

> Für dein Portfolio oder Kursprojekt solltest du die Datenquelle immer sauber dokumentieren: Herkunft, URL, Lizenz, Erhebungsmethode und Zeitraum.

---

### 5. Datentypen verstehen

Nicht alle Daten liegen in derselben Struktur vor.

#### Strukturierte Daten

Strukturierte Daten besitzen eine feste tabellarische Form.

Beispiele:

- CSV-Dateien

- Excel-Tabellen

- SQL-Datenbanken

- Tabellen mit klaren Spalten und Datentypen

Typische Verarbeitung:

```python

import pandas as pd

df = pd.read_csv("dataset.csv")

df.info()

df.describe()

```

#### Semi-strukturierte Daten

Semi-strukturierte Daten besitzen eine gewisse Ordnung, aber keine klassische Tabellenstruktur.

Beispiele:

- JSON

- XML

- NoSQL-Dokumente

- verschachtelte API-Antworten

Beispiel:

```json

{

"kunde": {

"id": 42,

"alter": 31,

"vertrag": "Premium"

}

}

```

#### Unstrukturierte Daten

Unstrukturierte Daten besitzen keine feste vorgegebene Struktur.

Beispiele:

- Texte

- PDFs

- Bilder

- Videos

- Audiodateien

>[!important]

> Unstrukturierte Daten müssen meistens erst transformiert werden, bevor klassische ML-Algorithmen damit arbeiten können.

>

> Text wird z. B. vektorisiert, Bilder werden in Pixelwerte oder Embeddings umgewandelt.

---

### 6. Ziel der Datenexploration

Die Datenexploration wird häufig auch **Exploratory Data Analysis (EDA)** genannt.

Sie verfolgt vier zentrale Ziele:

| Ziel | Erklärung |

| ---- | ---- |

| **Datenqualität prüfen** | Fehler, Inkonsistenzen, Duplikate und ungültige Werte erkennen |

| **Verteilungen analysieren** | Statistische Eigenschaften einzelner Variablen verstehen |

| **Ausreißer identifizieren** | Extreme oder ungewöhnliche Werte erkennen und bewerten |

| **Beziehungen erkennen** | Korrelationen und Zusammenhänge zwischen Variablen untersuchen |

>[!note]

> EDA ist kein optionaler Schönheitsfilter für Diagramme.

>

> EDA ist der Schritt, in dem man merkt, dass die Daten doch nicht so sauber sind, wie irgendein optimistischer CSV-Dateiname behauptet hat.

---

### 7. Grundlegende Analyse mit pandas

Ein erster Blick auf den Datensatz erfolgt meist mit pandas.

```python

import pandas as pd

# Daten laden

df = pd.read_csv("dataset.csv")

# Struktur anzeigen

print(df.info())

# Deskriptive Statistik

print(df.describe())

# Fehlende Werte pro Spalte

print(df.isnull().sum())

# Anteil fehlender Werte in Prozent

missing_pct = (df.isnull().sum() / len(df)) * 100

print(missing_pct)

```

Wichtige Funktionen:

| Funktion | Zweck |

| ---- | ---- |

| `info()` | zeigt Datentypen und Anzahl nicht-fehlender Werte |

| `describe()` | liefert deskriptive Statistiken |

| `isnull()` | markiert fehlende Werte |

| `sum()` | zählt fehlende Werte je Spalte |

| `value_counts()` | zählt Häufigkeiten kategorialer Werte |

| `corr()` | berechnet Korrelationen numerischer Merkmale |

---

### 8. Visualisierung strukturierter Daten

Visualisierungen helfen dabei, Muster zu erkennen, die in reinen Tabellen leicht übersehen werden.

| Visualisierung | Zweck | Geeignet für |

| ---- | ---- | ---- |

| **Histogramm** | Verteilung und Häufigkeit eines Merkmals darstellen | numerische Features |

| **Boxplot** | Ausreißer, Median, Quartile und Streuung erkennen | numerische Features |

| **Scatterplot** | Zusammenhang zwischen zwei Variablen untersuchen | zwei numerische Features |

| **Heatmap** | Korrelationen zwischen mehreren Merkmalen darstellen | numerische Feature-Matrix |

---

### 9. Histogramme: Verteilungen erkennen

Ein Histogramm zeigt, wie häufig bestimmte Wertebereiche vorkommen.

Beispiel:

```python

import seaborn as sns

import matplotlib.pyplot as plt

sns.histplot(data=df, x="Alter", bins=30, kde=True)

plt.title("Altersverteilung")

plt.show()

```

Worauf achten?

- Ist die Verteilung ungefähr normalverteilt?

- Ist sie links- oder rechtsschief?

- Gibt es mehrere Gipfel?

- Gibt es auffällige Lücken oder Häufungen?

>[!tip]

> Die KDE-Kurve glättet die Verteilung und macht Trends leichter sichtbar.

---

### 10. Boxplots: Ausreißer erkennen

Ein Boxplot fasst eine numerische Verteilung kompakt zusammen.

Er zeigt:

- Median

- erstes und drittes Quartil

- Interquartilsabstand

- typische Streuung

- potenzielle Ausreißer

```python

sns.boxplot(data=df, y="Einkommen")

plt.title("Einkommensverteilung und potenzielle Ausreißer")

plt.show()

```

Interpretation:

| Element | Bedeutung |

| ---- | ---- |

| Box | mittlere 50 % der Daten |

| Linie in der Box | Median |

| Whiskers | typische Streuung |

| Punkte außerhalb | potenzielle Ausreißer |

>[!important]

> Ein Ausreißer ist nicht automatisch ein Fehler.

>

> Ein sehr hohes Einkommen kann ein Datenfehler sein. Es kann aber auch einfach ein sehr reiches Individuum sein, weil die Realität leider so programmiert wurde.

---

### 11. Scatterplots: Zusammenhänge erkennen

Scatterplots zeigen den Zusammenhang zwischen zwei numerischen Variablen.

```python

sns.scatterplot(data=df, x="Alter", y="Einkommen")

plt.title("Zusammenhang zwischen Alter und Einkommen")

plt.show()

```

Mögliche Beobachtungen:

- positive Korrelation

- negative Korrelation

- keine erkennbare Beziehung

- Clusterbildung

- nichtlineare Zusammenhänge

- Ausreißer

Beispielhafte Interpretation:

```text

Wenn mit steigendem Alter tendenziell auch das Einkommen steigt,

liegt eine positive Korrelation vor.

```

---

### 12. Heatmaps: Korrelationen sichtbar machen

Eine Korrelations-Heatmap zeigt Beziehungen zwischen mehreren numerischen Merkmalen.

```python

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)

plt.title("Korrelationsmatrix aller numerischen Merkmale")

plt.show()

```

Interpretation:

| Korrelationswert | Bedeutung |

| ---- | ---- |

| nahe **+1** | starke positive Beziehung |

| nahe **-1** | starke negative Beziehung |

| nahe **0** | kaum linearer Zusammenhang |

>[!note]

> Korrelation bedeutet nicht Kausalität.

>

> Nur weil zwei Dinge zusammen auftreten, heißt das nicht, dass eines das andere verursacht. Sonst wäre Eisverkauf auch schuld an Sonnenbrand. Menschliche Statistikinterpretation bleibt ein Abenteuer.

---

### 13. Visualisierung semi-strukturierter und unstrukturierter Daten

Nicht nur tabellarische Daten können visualisiert werden.

#### Textdaten: Wordclouds

Wordclouds zeigen häufige Begriffe in Textdaten.

Beispielhafte Anwendung:

- Kundenbewertungen

- Social-Media-Kommentare

- Support-Tickets

- Freitextantworten in Umfragen

Häufigere Wörter erscheinen größer.

>[!warning]

> Wordclouds sind gut für einen schnellen Überblick, aber schlecht für präzise Analyse.

> Für ernsthafte Textanalyse sind zusätzliche Methoden wie Tokenisierung, TF-IDF oder Embeddings sinnvoller.

#### Bilddaten: Pixelwerte als Matrix

Bilder können als numerische Matrizen interpretiert werden.

Bei Graustufenbildern entspricht jeder Pixel einem Intensitätswert.

```text

Bild → Pixelmatrix → numerische Daten → Modellinput

```

Heatmaps können helfen, Muster in Pixelwerten sichtbar zu machen.

---

### 14. Fehlende Werte: Warum entstehen sie?

Fehlende Werte sind in echten Datensätzen sehr häufig.

Typische Ursachen:

| Ursache | Erklärung |

| ---- | ---- |

| **Erhebungsfehler** | technische Fehler, falsche Eingaben, Messprobleme |

| **Systembedingte Ausfälle** | Sensorfehler, Serverausfälle, Übertragungsprobleme |

| **Bewusste Nichtangaben** | Datenschutz, irrelevante Fragen, verweigerte Angaben |

>[!important]

> Fehlende Werte dürfen nicht einfach ignoriert werden.

>

> Sie können Analysen verzerren, Modelle beschädigen und zu falschen Entscheidungen führen.

---

### 15. Arten fehlender Werte: MCAR, MAR, MNAR

Um fehlende Werte sinnvoll zu behandeln, muss man verstehen, warum sie fehlen.

| Typ | Bedeutung | Beispiel |

| ---- | ---- | ---- |

| **MCAR** | Missing Completely At Random | Werte fehlen rein zufällig, z. B. durch technischen Zufallsausfall |

| **MAR** | Missing At Random | Fehlen hängt von beobachtbaren anderen Variablen ab |

| **MNAR** | Missing Not At Random | Fehlen hängt mit dem fehlenden Wert selbst zusammen |

Beispiele:

- **MCAR:** Einige Messwerte fehlen zufällig wegen eines kurzen Systemfehlers.

- **MAR:** Jüngere Kunden geben seltener ihr Einkommen an, aber das Alter ist bekannt.

- **MNAR:** Personen mit sehr hohem Einkommen geben ihr Einkommen bewusst nicht an.

>[!tip]

> Die Art des Fehlens beeinflusst, welche Imputationsmethode sinnvoll ist.

---

### 16. Folgen fehlender Werte

Fehlende Werte können mehrere Probleme verursachen.

| Folge | Erklärung |

| ---- | ---- |

| **Verzerrte Analysen** | Kennzahlen wie Mittelwert oder Standardabweichung werden verfälscht |

| **Modellfehler** | Viele ML-Algorithmen können mit `NaN` nicht direkt umgehen |

| **Schlechtere Performance** | Informationsverlust reduziert die Vorhersagequalität |

| **Bias** | systematisch fehlende Werte können bestimmte Gruppen benachteiligen |

>[!warning]

> Wenn fehlende Werte systematisch auftreten, ist das Problem nicht nur technisch, sondern analytisch.

> Dann reicht ein schnelles `fillna()` nicht aus, auch wenn es sich schön nach Produktivität anfühlt.

---

### 17. Fehlende Werte analysieren

Vor der Imputation sollte man fehlende Werte sichtbar machen.

```python

# Anzahl fehlender Werte je Spalte

missing_count = df.isnull().sum()

# Prozentualer Anteil

missing_pct = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({

"missing_count": missing_count,

"missing_pct": missing_pct

})

print(missing_summary.sort_values("missing_pct", ascending=False))

```

Mögliche Leitfragen:

- Welche Features haben fehlende Werte?

- Wie hoch ist der Anteil pro Feature?

- Fehlen Werte zufällig oder systematisch?

- Gibt es Gruppen, bei denen besonders viele Werte fehlen?

- Soll ein Feature entfernt, imputiert oder speziell markiert werden?

---

### 18. Methoden zur Behandlung fehlender Werte

Es gibt verschiedene Strategien.

| Methode | Idee | Geeignet für | Risiko |

| ---- | ---- | ---- | ---- |

| **Zeilen löschen** | Entfernt Datenpunkte mit fehlenden Werten | sehr geringe Fehlerrate | Informationsverlust |

| **Spalten löschen** | Entfernt ganze Features | sehr viele fehlende Werte | wichtiges Feature kann verloren gehen |

| **Mittelwert-Imputation** | ersetzt fehlende Werte durch Mittelwert | numerische Daten, MCAR | verzerrt Verteilung |

| **Median-Imputation** | ersetzt durch Median | numerische Daten mit Ausreißern | reduziert Varianz |

| **Modus-Imputation** | ersetzt durch häufigsten Wert | kategoriale Daten | kann dominante Klasse verstärken |

| **KNN-Imputation** | nutzt ähnliche Datenpunkte | korrelierte Features | rechenintensiver |

| **MICE** | multiple iterative Imputation | MAR-Daten | komplexer |

| **MissForest** | Random-Forest-basierte Imputation | gemischte Datentypen | hoher Rechenaufwand |

---

### 19. Einfache Imputation mit pandas

Beispiel: Mittelwert-Imputation für eine numerische Spalte.

```python

# Fehlende Werte zählen

print(df["Einkommen"].isnull().sum())

# Mittelwert berechnen

mean_income = df["Einkommen"].mean()

# Fehlende Werte ersetzen

df["Einkommen"] = df["Einkommen"].fillna(mean_income)

# Ergebnis prüfen

print(df["Einkommen"].isnull().sum())

```

>[!warning]

> Die Mittelwert-Imputation ist einfach, aber nicht automatisch gut.

> Sie kann Verteilungen glätten, Varianz reduzieren und Zusammenhänge abschwächen.

---

### 20. Imputation mit scikit-learn: SimpleImputer

```python

from sklearn.impute import SimpleImputer

# Mittelwert-Imputation

imputer_mean = SimpleImputer(strategy="mean")

df[["Alter", "Einkommen"]] = imputer_mean.fit_transform(

df[["Alter", "Einkommen"]]

)

# Median-Imputation

imputer_median = SimpleImputer(strategy="median")

# Modus-Imputation für kategoriale Daten

imputer_mode = SimpleImputer(strategy="most_frequent")

```

>[!tip]

> Für numerische Daten mit Ausreißern ist der Median oft robuster als der Mittelwert.

---

### 21. Fortgeschrittene Imputation: KNNImputer

Die KNN-Imputation ersetzt fehlende Werte anhand ähnlicher Datenpunkte.

```python

from sklearn.impute import KNNImputer

import pandas as pd

knn_imputer = KNNImputer(n_neighbors=5)

df_imputed = knn_imputer.fit_transform(df)

df = pd.DataFrame(df_imputed, columns=df.columns)

```

Vorteile:

- berücksichtigt Beziehungen zwischen Features

- nutzt Informationen ähnlicher Datenpunkte

- erzeugt oft realistischere Werte als einfache Imputation

Nachteile:

- benötigt numerische oder passend codierte Daten

- rechenintensiver bei großen Datensätzen

- empfindlich gegenüber unskalierten Daten

>[!important]

> KNN-Imputation sollte meistens nach geeigneter Vorbereitung numerischer Features eingesetzt werden, weil Distanzberechnungen sonst von großen Wertebereichen dominiert werden können.

---

### 22. Missing Indicator Feature

Manchmal ist nicht nur der Wert wichtig, sondern auch die Tatsache, dass ein Wert fehlt.

Beispiel:

```python

df["Einkommen_missing"] = df["Einkommen"].isnull().astype(int)

```

Das kann sinnvoll sein, wenn fehlende Werte systematisch auftreten.

Beispiel:

```text

Kunden ohne Zufriedenheitswert könnten weniger Kontakt zum Unternehmen haben.

Das Fehlen selbst enthält dann Information.

```

---

### 23. Skalierung und Normalisierung: Warum überhaupt?

Viele ML-Algorithmen reagieren empfindlich darauf, wenn Features sehr unterschiedliche Wertebereiche haben.

Beispiel:

| Feature | Wertebereich |

| ---- | ---- |

| Alter | 18 bis 90 |

| Einkommen | 20.000 bis 500.000 |

Ohne Skalierung dominiert das Einkommen Distanzberechnungen, obwohl Alter ebenfalls relevant sein könnte.

Besonders betroffen sind:

- k-Nearest Neighbors

- k-Means

- Support Vector Machines

- neuronale Netze

- PCA

- lineare und logistische Regressionen mit Regularisierung

Weniger empfindlich sind häufig:

- Decision Trees

- Random Forests

- Gradient Boosting Modelle

>[!note]

> Baumverfahren teilen Daten anhand von Schwellenwerten auf und sind daher oft robuster gegenüber unterschiedlichen Skalen.

> Distanzbasierte Verfahren dagegen leiden sofort, wenn ein Feature zahlenmäßig den Rest anschreit.

---

### 24. Skalierung vs. Normalisierung

Die Begriffe werden oft ähnlich verwendet, bedeuten aber nicht exakt dasselbe.

| Begriff | Bedeutung |

| ---- | ---- |

| **Skalierung** | Oberbegriff für Transformationen auf vergleichbare Wertebereiche |

| **Standardisierung** | Transformation auf Mittelwert 0 und Standardabweichung 1 |

| **Normalisierung** | häufig Transformation in den Bereich `[0, 1]` oder `[-1, 1]` |

>[!important]

> In der Praxis ist weniger der Begriff entscheidend, sondern die konkrete Methode und deren Wirkung auf die Daten.

---

### 25. Standardisierung: Z-Score-Transformation

Die Standardisierung transformiert Werte so, dass die neue Verteilung einen Mittelwert von 0 und eine Standardabweichung von 1 besitzt.

```text

z = (x - μ) / σ

```

| Symbol | Bedeutung |

| ---- | ---- |

| `x` | ursprünglicher Wert |

| `μ` | Mittelwert |

| `σ` | Standardabweichung |

| `z` | standardisierter Wert |

Eigenschaften:

- Mittelwert wird 0

- Standardabweichung wird 1

- ursprüngliche Verteilungsform bleibt erhalten

- Wertebereich ist nicht begrenzt

Geeignet für:

- lineare Regression

- logistische Regression

- PCA

- SVM

- k-NN

- normalverteilte oder annähernd normalverteilte Daten

Beispiel:

```python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

```

---

### 26. Min-Max-Skalierung

Die Min-Max-Skalierung transformiert Werte in einen festen Bereich, meistens `[0, 1]`.

```text

x_scaled = (x - x_min) / (x_max - x_min)

```

Eigenschaften:

- Werte liegen zwischen 0 und 1

- relative Abstände bleiben proportional erhalten

- einfach interpretierbar

- empfindlich gegenüber Ausreißern

Geeignet für:

- neuronale Netze

- k-NN

- k-Means

- Daten mit bekanntem Wertebereich

Beispiel:

```python

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

```

>[!warning]

> Ein einzelner extremer Ausreißer kann die Min-Max-Skalierung stark verzerren.

---

### 27. Logarithmische Skalierung

Die logarithmische Skalierung eignet sich besonders für stark rechtsschiefe Verteilungen.

```text

x_log = log(x + 1)

```

Die Addition von 1 verhindert Probleme bei `log(0)`.

Typische Anwendungen:

- Einkommen

- Preise

- Website-Traffic

- Bevölkerungszahlen

- Transaktionsbeträge

Beispiel:

```python

import numpy as np

df["Einkommen_log"] = np.log1p(df["Einkommen"])

```

Vorteile:

- reduziert Einfluss extremer Werte

- macht rechtsschiefe Verteilungen symmetrischer

- kann Modellperformance verbessern

---

### 28. Welche Skalierungsmethode wann?

| Situation | Geeignete Methode |

| ---- | ---- |

| normalverteilte Daten | Standardisierung |

| distanzbasierte Algorithmen | Standardisierung oder Min-Max-Skalierung |

| neuronale Netze | Min-Max-Skalierung oder Standardisierung |

| stark rechtsschiefe Daten | Log-Skalierung oder Box-Cox-Transformation |

| viele Ausreißer | robuste Skalierung oder Log-Transformation |

| baumbasierte Modelle | oft keine Skalierung notwendig |

>[!tip]

> Skalierung ist kein Selbstzweck.

> Entscheidend ist, welcher Algorithmus verwendet wird und welche Verteilung die Daten haben.

---

### 29. Data Leakage bei Skalierung vermeiden

Ein sehr häufiger Fehler besteht darin, Skalierer auf Testdaten erneut anzupassen.

#### Falsch

```python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

# FALSCH: fit_transform auf Testdaten

X_test_scaled = scaler.fit_transform(X_test)

```

#### Richtig

```python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit nur auf Trainingsdaten

X_train_scaled = scaler.fit_transform(X_train)

# Testdaten nur transformieren

X_test_scaled = scaler.transform(X_test)

```

Warum?

Die Parameter des Skalierers, z. B. Mittelwert und Standardabweichung, dürfen nur aus den Trainingsdaten gelernt werden.

>[!danger]

> Wenn Testdaten beim Fitten verwendet werden, fließen Informationen aus den Testdaten in den Trainingsprozess ein.

> Die Evaluation wirkt dann besser, als sie in der Realität ist. Das Modell betrügt nicht. Der Mensch hat nur die Versuchsanordnung ruiniert.

---

### 30. Preprocessing mit Pipeline

In der Praxis sollte Preprocessing möglichst in einer Pipeline organisiert werden.

```python

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([

("imputer", SimpleImputer(strategy="median")),

("scaler", StandardScaler()),

("model", KNeighborsClassifier(n_neighbors=5))

])

pipeline.fit(X_train, y_train)

score = pipeline.score(X_test, y_test)

print(score)

```

Vorteile:

- weniger Data Leakage

- besser reproduzierbar

- sauberere Struktur

- leichter mit Cross-Validation kombinierbar

- besser für spätere Produktion geeignet

---

### 31. Beispielworkflow für die U-Phase

Ein sinnvoller Ablauf für „Understanding the Data“ kann so aussehen:

```text

1. Datensatz laden

2. Datenquelle und Lizenz dokumentieren

3. Datenstruktur prüfen

4. Datentypen analysieren

5. Zielvariable bestimmen

6. Fehlende Werte untersuchen

7. Duplikate und Inkonsistenzen prüfen

8. Deskriptive Statistik berechnen

9. Visualisierungen erstellen

10. Ausreißer bewerten

11. Korrelationen untersuchen

12. Imputationsstrategie festlegen

13. Skalierungsmethode auswählen

14. Preprocessing dokumentieren

15. vorbereiteten Datensatz für Modellierung speichern

```

---

### 32. Praktische Übung zur Einheit

#### Aufgabe 1: Datensatz auswählen und beschreiben

Wähle einen geeigneten Datensatz, z. B. von:

- Kaggle

- UCI Machine Learning Repository

- Google Dataset Search

- OpenML

- Data.gov

- European Data Portal

Dokumentiere:

| Punkt | Inhalt |

| ---- | ---- |

| Herkunft | URL und Plattform |

| Erhebungsmethode | Wie wurden die Daten gesammelt? |

| Zeitraum | Wann wurden die Daten erfasst? |

| Lizenz | Darf der Datensatz verwendet werden? |

| Struktur | Zeilen, Spalten, Datentypen |

| Zielvariable | Was soll vorhergesagt oder untersucht werden? |

---

#### Aufgabe 2: Erste Visualisierungen erstellen

Erstelle mindestens:

- drei Histogramme für numerische Features

- Boxplots für numerische Features

- Scatterplots für relevante Feature-Paare

- eine Korrelations-Heatmap

Analysiere:

- Verteilungen

- Schiefe

- Ausreißer

- lineare oder nichtlineare Zusammenhänge

- stark korrelierte Features

---

#### Aufgabe 3: Fehlende Werte untersuchen und imputieren

Bearbeite folgende Punkte:

1. Fehlende Werte je Feature zählen

2. Prozentualen Anteil fehlender Werte berechnen

3. Muster fehlender Werte interpretieren

4. Mindestens zwei Imputationsmethoden anwenden

5. Ergebnisse vergleichen

6. Entscheidung begründen

Mögliche Methoden:

- Median-Imputation

- Mittelwert-Imputation

- Modus-Imputation

- KNN-Imputation

- Missing Indicator

>[!tip]

> Erstelle für verschiedene Imputationsstrategien separate Versionen des Datensatzes.

> Dann kannst du die Auswirkungen direkt vergleichen, statt im Blindflug an `df_final_final_v3_really_final.csv` herumzuschrauben.

---

#### Aufgabe 4: Numerische Merkmale skalieren

Vorgehen:

1. Verteilungen prüfen

2. passende Skalierungsmethode wählen

3. Skalierung mit scikit-learn implementieren

4. Vorher-Nachher-Vergleich visualisieren

5. Auswirkungen auf spätere ML-Algorithmen erklären

Beispiel:

```python

from sklearn.preprocessing import StandardScaler, MinMaxScaler

numeric_features = ["Alter", "Einkommen"]

standard_scaler = StandardScaler()

minmax_scaler = MinMaxScaler()

X_standardized = standard_scaler.fit_transform(df[numeric_features])

X_normalized = minmax_scaler.fit_transform(df[numeric_features])

```

---

#### Aufgabe 5: Jupyter-Notebook dokumentieren

Das Notebook sollte folgende Abschnitte enthalten:

| Abschnitt | Inhalt |

| ---- | ---- |

| **Datensatz-Übersicht** | Quelle, Struktur, Features, Zielvariable |

| **Explorative Analyse** | zentrale Visualisierungen und Erkenntnisse |

| **Datenqualität** | fehlende Werte, Ausreißer, Inkonsistenzen |

| **Preprocessing** | Imputation, Skalierung, Begründungen |

| **Zusammenfassung** | wichtigste Ergebnisse und Schlussfolgerungen |

| **Reflexion** | Herausforderungen und nächste Schritte |

>[!important]

> Das Notebook sollte Code, Visualisierungen und erklärende Markdown-Zellen enthalten.

>

> Nur Code ohne Erklärung ist keine Analyse. Es ist ein digitales Rätselheft.

---

### 33. Praxisnahe Fallstudien

#### Fallbeispiel 1: Kreditrisikobewertung

Problem:

Ein Finanzinstitut möchte das Kreditrisiko von Kunden vorhersagen.

Herausforderungen:

- Einkommen ist stark rechtsschief verteilt

- einzelne sehr hohe Einkommen verzerren die Daten

- fehlende Werte beeinträchtigen die Modellqualität

Mögliche Lösung:

- Log-Skalierung für Einkommen

- KNN-Imputation für fehlende Werte

- Vergleich der Modellleistung vor und nach Preprocessing

Erkenntnis:

>[!quote]

> Eine passende Transformation kann entscheidend sein, damit ein Modell relevante Muster erkennt, statt nur große Zahlen zu bewundern.

---

#### Fallbeispiel 2: Kundenabwanderung / Churn Prediction

Problem:

Ein Telekommunikationsunternehmen möchte vorhersagen, welche Kunden kündigen werden.

Herausforderung:

- Das Feature „Kundenzufriedenheit“ hat viele fehlende Werte.

- Fehlende Werte treten nicht zufällig auf.

- Kunden ohne Zufriedenheitswert hatten möglicherweise weniger Kontakt zum Unternehmen.

Mögliche Lösung:

- Analyse des Missing-Patterns

- Missing Indicator Feature

- KNN-Imputation anhand ähnlicher Kundenprofile

- Korrelationsanalyse mit Vertragslaufzeit, Serviceanfragen und Kündigung

Erkenntnis:

>[!quote]

> Manchmal ist das Fehlen eines Wertes selbst ein wichtiges Signal.

---

#### Fallbeispiel 3: Bildklassifikation

Problem:

Ein Deep-Learning-Modell soll Produktbilder klassifizieren.

Ausgangslage:

- Pixelwerte liegen im Bereich `[0, 255]`

- unskalierte Werte erschweren das Training

Preprocessing:

1. Pixelwerte auf `[0, 1]` skalieren

2. Standardisierung mit Mittelwert und Standardabweichung

3. Data Augmentation zur Erhöhung der Robustheit

Erkenntnis:

>[!quote]

> Bei neuronalen Netzen ist Skalierung oft entscheidend für schnelle und stabile Konvergenz.

---

### 34. Häufige Fehler und Best Practices

#### Fehler 1: Fehlende Werte ignorieren

Problem:

- Modelle können mit fehlenden Werten oft nicht umgehen.

- Analysen werden verzerrt.

- systematische Fehlmuster bleiben unentdeckt.

Besser:

- fehlende Werte immer analysieren

- Missing-Pattern prüfen

- passende Imputationsstrategie begründen

---

#### Fehler 2: Data Leakage durch falsches Preprocessing

Problem:

- Skalierer oder Imputer werden auf Testdaten gefittet.

- Testinformationen fließen indirekt ins Modell ein.

- Evaluation wird unrealistisch gut.

Besser:

```python

# richtiges Prinzip

preprocessing.fit(X_train)

X_train_processed = preprocessing.transform(X_train)

X_test_processed = preprocessing.transform(X_test)

```

---

#### Fehler 3: Falsche Visualisierung wählen

| Falsch | Besser |

| ---- | ---- |

| Histogramm für kategoriale Daten | Balkendiagramm |

| Kuchendiagramm mit vielen Kategorien | Balkendiagramm |

| 3D-Diagramm ohne Mehrwert | klares 2D-Diagramm |

| zu viele Farben | reduzierte Farbpalette |

>[!tip]

> Die Visualisierung sollte zur Fragestellung passen, nicht zum Bedürfnis, möglichst wissenschaftlich auszusehen.

---

### 35. Best Practice: Dokumentation

Eine gute Datenanalyse ist reproduzierbar.

Dokumentiere deshalb:

- Datenquelle

- Datensatzversion

- Anzahl Zeilen und Spalten

- Bedeutung der Features

- Zielvariable

- fehlende Werte

- erkannte Ausreißer

- verwendete Imputationsmethode

- verwendete Skalierungsmethode

- Begründung jeder Preprocessing-Entscheidung

- mögliche Risiken und Annahmen

Geeignete Werkzeuge:

- Jupyter Notebook

- Git / GitHub

- README.md

- requirements.txt

- MLFlow

- Data Sheets

- Model Cards

---

### 36. Verbindung zu MLOps

Die U-Phase sollte so dokumentiert werden, dass spätere Experimente reproduzierbar sind.

| Aufgabe | MLOps-Bezug |

| ---- | ---- |

| Datenversion speichern | Reproduzierbarkeit |

| Preprocessing dokumentieren | Vergleichbarkeit |

| Pipelines nutzen | weniger manuelle Fehler |

| Parameter speichern | Nachvollziehbarkeit |

| Notebooks versionieren | Teamfähigkeit |

>[!important]

> Ein Preprocessing-Schritt, der nicht dokumentiert ist, existiert praktisch nicht zuverlässig.

>

> Oder schlimmer: Er existiert irgendwo in einer Notebook-Zelle, die nie wieder jemand in der richtigen Reihenfolge ausführt.

---

### 37. Checkliste für Einheit 2

>[!check]

> Für dein eigenes Projekt solltest du nach dieser Einheit folgende Punkte abhaken können:

- [ ] Forschungsfrage oder Analyseziel ist klar

- [ ] Datensatzquelle ist dokumentiert

- [ ] Lizenz und Nutzung sind geklärt

- [ ] Zeilen, Spalten und Datentypen sind bekannt

- [ ] Zielvariable ist definiert

- [ ] numerische und kategoriale Features sind getrennt

- [ ] fehlende Werte wurden analysiert

- [ ] Ausreißer wurden untersucht

- [ ] zentrale Visualisierungen wurden erstellt

- [ ] Korrelationen wurden geprüft

- [ ] Imputationsstrategie wurde ausgewählt und begründet

- [ ] Skalierungsmethode wurde ausgewählt und begründet

- [ ] Data Leakage wurde vermieden

- [ ] Notebook enthält erklärende Markdown-Zellen

- [ ] Ergebnisse sind reproduzierbar dokumentiert

---

### 38. Zentrale Begriffe

| Begriff | Kurzdefinition |

| ---- | ---- |

| **EDA** | Explorative Datenanalyse zur Untersuchung von Struktur, Qualität und Mustern |

| **Feature** | Eingabevariable eines Modells |

| **Target / Label** | Zielvariable, die vorhergesagt werden soll |

| **Imputation** | Ersetzen fehlender Werte durch geschätzte oder berechnete Werte |

| **KNN-Imputation** | Imputation anhand ähnlicher Datenpunkte |

| **MICE** | Multiple Imputation by Chained Equations |

| **MissForest** | Random-Forest-basierte Imputation |

| **Skalierung** | Transformation von Features auf vergleichbare Wertebereiche |

| **Standardisierung** | Transformation auf Mittelwert 0 und Standardabweichung 1 |

| **Normalisierung** | Transformation in einen festen Bereich, häufig `[0, 1]` |

| **Data Leakage** | Informationsfluss aus Testdaten in Training oder Preprocessing |

| **Korrelationsmatrix** | Matrix paarweiser Korrelationen zwischen numerischen Variablen |

| **Outlier** | Datenpunkt, der deutlich von der Mehrheit der Werte abweicht |

| **Pipeline** | Verkettung von Preprocessing- und Modellierungsschritten |

---

### 39. Merksätze

> [!quote]

> Die U-Phase verhindert, dass blind modelliert wird.

> [!quote]

> EDA macht sichtbar, welche Probleme und Muster in den Daten stecken.

> [!quote]

> Fehlende Werte sind nicht nur Lücken, sondern oft Hinweise auf den Datenerhebungsprozess.

> [!quote]

> Ausreißer müssen bewertet werden, bevor sie entfernt werden.

> [!quote]

> Skalierung ist besonders wichtig für distanzbasierte Algorithmen und neuronale Netze.

> [!quote]

> `fit()` gehört auf Trainingsdaten, `transform()` auf Trainings- und Testdaten.

> [!quote]

> Dokumentation ist kein Bonus, sondern Teil der Analyse.

---

### 40. Prüfungs- und Verständnisfragen

1. Warum ist die U-Phase im QUA³CK-Modell so wichtig?

2. Welche vier Ziele verfolgt die Datenexploration?

3. Was ist der Unterschied zwischen strukturierten, semi-strukturierten und unstrukturierten Daten?

4. Welche Visualisierung eignet sich für die Verteilung eines numerischen Features?

5. Wofür verwendet man Boxplots?

6. Warum ist Korrelation nicht automatisch Kausalität?

7. Welche Ursachen können fehlende Werte haben?

8. Was ist der Unterschied zwischen MCAR, MAR und MNAR?

9. Wann ist Mittelwert-Imputation problematisch?

10. Warum kann KNN-Imputation bessere Ergebnisse liefern als einfache Imputation?

11. Warum müssen manche Features skaliert werden?

12. Was ist der Unterschied zwischen Standardisierung und Min-Max-Skalierung?

13. Für welche Algorithmen ist Skalierung besonders wichtig?

14. Warum dürfen Skalierer nicht auf Testdaten gefittet werden?

15. Was ist Data Leakage?

16. Welche Abschnitte sollte ein gutes Jupyter Notebook zur Datenanalyse enthalten?

17. Warum sind Pipelines in scikit-learn sinnvoll?

18. Welche Rolle spielt Dokumentation für MLOps?

---

### 41. Mini-Zusammenfassung

Die Einheit behandelt die **Datenphase im QUA³CK-Modell**.

In der U-Phase werden Datenquellen, Datentypen, Datenqualität, fehlende Werte, Ausreißer, Verteilungen und Zusammenhänge untersucht.

Zentrale Methoden sind:

- deskriptive Statistik

- Histogramme

- Boxplots

- Scatterplots

- Korrelations-Heatmaps

- Imputation fehlender Werte

- Feature Scaling

- Normalisierung und Standardisierung

Besonders wichtig ist, Preprocessing-Schritte sauber zu dokumentieren und **Data Leakage** zu vermeiden.

>[!important]

> Gute Modelle entstehen nicht durch blindes Ausprobieren von Algorithmen, sondern durch systematisches Verständnis der Daten.

---

### Aufgabe

>[!important]

> Erstelle für dein eigenes QUA³CK-Projekt ein Jupyter Notebook zur Phase **Understanding the Data**.

>

> Das Notebook soll folgende Punkte enthalten:

>

> - Datensatz laden und Quelle dokumentieren

> - Datenstruktur mit `info()` und `describe()` untersuchen

> - numerische und kategoriale Features identifizieren

> - Zielvariable bestimmen

> - fehlende Werte zählen und prozentual darstellen

> - mindestens drei Visualisierungstypen einsetzen

> - Ausreißer analysieren

> - Korrelationen untersuchen

> - mindestens zwei Imputationsmethoden vergleichen

> - Skalierungsmethode auswählen und begründen

> - Data Leakage vermeiden

> - Ergebnisse in Markdown-Zellen interpretieren

>

> Bonus:

>

> - Nutze eine scikit-learn `Pipeline`

> - Speichere relevante Plots im Repo

> - Dokumentiere deine Entscheidungen im README
