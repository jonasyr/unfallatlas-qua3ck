---
title: Data Analytics und Big Data
description: ""
date: 05-05-2026
time: 10:04
index: "[[Studium]]"
subindex: ""
category:
  - source
status:
  - begin
---

# Data Analytics und Big Data

>- **Index:** [[Studium]]
>- **Document Tags:**

## Notizen

```dataviewjs
const pages = dv.pages('"6 - Notes"')
  .where(p => p.reference == "Data Analytics und Big Data")
  .sort(p => p.file.name);

dv.table(["Notes", "Description", "Date", "Status"], 
  pages.map(p => [
    p.file.link,
    p.description,
    p.date,
    (() => {
      if (p.status) {
        if (p.status.includes("begin") && p.status.includes("finish")) {
          return "<span style='color:purple'>not Initialized</span>";
        } else if (p.status.includes("begin")) {
          return "<span style='color:red'>Begin</span>";
        } else if (p.status.includes("finish")) {
          return "<span style='color:green'>Finish</span>";
        }
      }
      return "<span style='color:purple'>not Initialized</span>";
    })()
  ])
)
```

---

## Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte

>[!summary]
> Diese Einheit führt das **QUA³CK-Prozessmodell** als strukturierten Rahmen für Machine-Learning- und Data-Science-Projekte ein.
> 
> Der Fokus liegt darauf, ein ML-Projekt nicht nur technisch umzusetzen, sondern von der **Fragestellung** über die **Datenanalyse** und **Modellentwicklung** bis zur **produktiven Anwendung** systematisch zu planen.

---

### 1. Warum braucht man strukturierte ML-Prozesse?

Viele Data-Science-Projekte scheitern nicht an fehlenden Algorithmen, sondern an fehlender Struktur.

Typische Probleme sind:

- unklare Problemdefinition
- fehlende Erfolgsmetriken
- keine saubere Dokumentation
- Experimente sind nicht reproduzierbar
- Modelle bleiben im Notebook und kommen nie in Produktion

>[!important]
> Ein gutes ML-Projekt beginnt nicht mit Code, sondern mit einer klaren Fragestellung.

Das QUA³CK-Modell hilft dabei, Data-Science-Projekte systematisch von der Idee bis zur Anwendung umzusetzen.

---

### 2. Lernziele der Einheit

Nach dieser Einheit solltest du:

- das **QUA³CK-Modell** als Prozess für ML-Projekte erklären können
- die fünf Phasen **Q, U, A³, C und K** unterscheiden können
- verstehen, warum **MLOps** für Reproduzierbarkeit und Qualität wichtig ist
- den Prozess am Beispiel eines **Iris-Klassifikators** anwenden können
- den Übergang von Analyse zu Deployment nachvollziehen können

---

### 3. Das QUA³CK-Modell im Überblick

Das **QUA³CK-Modell** wurde als praxisorientiertes Prozessmodell für Machine-Learning-Projekte entwickelt.

Es verbindet wissenschaftliches Vorgehen mit praktischer Umsetzbarkeit.

QUA³CK besteht aus fünf Phasen:

| Phase  | Bedeutung                                                         | Kernfrage                                               |
| ------ | ----------------------------------------------------------------- | ------------------------------------------------------- |
| **Q**  | Question                                                          | Welches Problem soll gelöst werden?                     |
| **U**  | Understanding the Data                                            | Welche Struktur und Qualität haben die Daten?           |
| **A³** | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | Welches Modell funktioniert wie gut?                    |
| **C**  | Conclude & Compare                                                | Welches Modell ist insgesamt am besten geeignet?        |
| **K**  | Knowledge Transfer                                                | Wie werden Ergebnisse dokumentiert und nutzbar gemacht? |


>[!note]
> Der Prozess ist nicht streng linear. Besonders die **A³-Phase** ist iterativ: Modelle werden trainiert, angepasst, bewertet und verbessert.
> 
> Gerade für die **U-Phase** ist [[Master Data Analysis with ChatGPT]] sehr praktisch

---

### 4. Q – Question: Problem verstehen

Die erste Phase ist die wichtigste Grundlage des gesamten Projekts.

In **Q** wird festgelegt:

- welches konkrete Problem gelöst werden soll
- wer die Zielgruppe ist
- welche Erfolgsmetriken gelten
- welches Ergebnis oder Produkt entstehen soll

#### Beispiel: Iris-Projekt

| Aspekt          | Beispiel                                   |
| --------------- | ------------------------------------------ |
| Problem         | Automatische Klassifikation von Iris-Arten |
| Zielgruppe      | Botanik-Studierende oder Nutzer im Feld    |
| Datenbasis      | Blütenmerkmale des Iris-Datensatzes        |
| Erfolgsmetrik   | Accuracy > 95 %                            |
| Deployment-Ziel | Interaktive Streamlit-App                  |

>[!important]
> Ohne klare Fragestellung kann ein technisch gutes Modell trotzdem fachlich nutzlos sein.

---

### 5. U – Understanding the Data: Daten verstehen

In der U-Phase werden die Daten untersucht, bevor ein Modell trainiert wird.

Ziele der Phase:

- Datenstruktur verstehen
- fehlende oder auffällige Werte erkennen
- Verteilungen analysieren
- Zusammenhänge zwischen Features erkennen
- erste Hypothesen für die Modellierung ableiten

Typische Methoden:

- deskriptive Statistik
- Scatterplots
- Boxplots
- Korrelationsanalyse
- Prüfung von Klassenverteilungen

#### Beispiel: Iris-Datensatz

Der Iris-Datensatz enthält:

| Bestandteil   | Beschreibung                                         |
| ------------- | ---------------------------------------------------- |
| Beobachtungen | 150 Iris-Blüten                                      |
| Klassen       | Setosa, Versicolor, Virginica                        |
| Features      | sepal length, sepal width, petal length, petal width |
| Zielvariable  | Iris-Art                                             |

Zentrale Erkenntnis aus der EDA:

- **Petal length** und **petal width** trennen die Iris-Arten deutlich besser als die Sepal-Merkmale.
- Setosa ist gut separierbar.
- Versicolor und Virginica überlappen stärker.

>[!tip]
> EDA entscheidet oft darüber, welche Features wichtig sind und welche Modelle sinnvoll sein könnten.

---

### 6. A³ – Algorithmen entwickeln und optimieren

Die A³-Phase ist die eigentliche Modellierungsphase.

A³ steht für:

| Bestandteil                   | Bedeutung                             |
| ----------------------------- | ------------------------------------- |
| **Algorithm Selection**       | geeignete Algorithmen auswählen       |
| **Adapting Features**         | Features anpassen oder transformieren |
| **Adjusting Hyperparameters** | Hyperparameter optimieren             |

Diese Phase wird mehrfach durchlaufen. Ziel ist es, verschiedene Modelle systematisch zu testen und zu verbessern.

#### Beispiel: „Big 3“ im Iris-Projekt

Im Kurs werden beispielhaft drei Ansätze verglichen:

| Modell              | Typ                  | Idee                                     |
| ------------------- | -------------------- | ---------------------------------------- |
| Decision Tree       | überwachtes Lernen   | Entscheidungsregeln als Baum             |
| K-Nearest Neighbors | überwachtes Lernen   | Klassifikation anhand ähnlicher Nachbarn |
| K-Means             | unüberwachtes Lernen | Gruppierung ähnlicher Datenpunkte        |

---

### 7. Wichtiger Unterschied: X/y-Split vs. Train/Test-Split

Diese beiden Begriffe werden oft verwechselt.

#### X/y-Split

Beim **X/y-Split** werden Features und Zielvariable getrennt.

| Symbol | Bedeutung                   |
| ------ | --------------------------- |
| **X**  | Eingabevariablen / Features |
| **y**  | Zielvariable / Label        |

Beispiel:

```python

X = df.drop("species", axis=1)

y = df["species"]

````

#### Train/Test-Split

Beim **Train/Test-Split** werden die Datenpunkte in Trainings- und Testdaten aufgeteilt.

| Datenteil      | Zweck                             |
| -------------- | --------------------------------- |
| Trainingsdaten | Modell lernt daraus               |
| Testdaten      | unabhängige Bewertung des Modells |

Beispiel:

```python

X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size=0.2, random_state=42

)

```

> [!note]
> **X/y-Split** trennt Spalten nach Rolle.
> **Train/Test-Split** trennt Zeilen nach Verwendung.

---

### 8. MLOps: Experimente reproduzierbar machen

MLOps steht für **Machine Learning Operations**.

Ziel von MLOps:

* Experimente nachvollziehbar dokumentieren
* Parameter und Metriken speichern
* Modelle systematisch vergleichen
* Deployment vorbereiten
* Zusammenarbeit erleichtern

Im Kurs wird dafür besonders **MLFlow** relevant.

Mit MLFlow können protokolliert werden:

* verwendete Parameter
* erreichte Metriken
* Modellversionen
* Artefakte
* Experimentläufe

> [!important]
> MLOps sorgt dafür, dass ein ML-Projekt nicht nur einmal funktioniert, sondern reproduzierbar, vergleichbar und weiterentwickelbar bleibt.

---

### 9. C – Conclude & Compare: Modelle bewerten

In der C-Phase werden die Ergebnisse der Experimente verglichen.

Dabei zählt nicht nur die höchste Accuracy. Ein Modell muss auch praktisch sinnvoll sein.

#### Quantitative Kriterien

| Metrik       | Bedeutung                                              |
| ------------ | ------------------------------------------------------ |
| Accuracy     | Anteil korrekter Vorhersagen                           |
| Precision    | Wie viele positive Vorhersagen waren korrekt?          |
| Recall       | Wie viele tatsächliche positive Fälle wurden gefunden? |
| F1-Score     | Ausgleich zwischen Precision und Recall                |
| Inferenzzeit | Geschwindigkeit einer Vorhersage                       |

#### Qualitative Kriterien

| Kriterium            | Bedeutung                                |
| -------------------- | ---------------------------------------- |
| Interpretierbarkeit  | Kann man Modellentscheidungen verstehen? |
| Komplexität          | Wie aufwendig ist das Modell?            |
| Wartbarkeit          | Wie leicht ist es später anzupassen?     |
| Deployment-Fähigkeit | Lässt es sich gut produktiv einsetzen?   |

#### Beispielergebnis im Iris-Projekt

| Modell              | Metrik              | Ergebnis |
| ------------------- | ------------------- | -------- |
| Decision Tree       | Accuracy            | 97,8 %   |
| K-Nearest Neighbors | Accuracy            | 97,8 %   |
| K-Means             | Adjusted Rand Score | 0,669    |

Obwohl Decision Tree und KNN gleich gut abschneiden, kann der **Decision Tree** bevorzugt werden, weil er interpretierbarer und effizienter sein kann.

> [!tip]
> Das beste Modell ist nicht automatisch das komplexeste Modell, sondern das Modell mit dem besten Verhältnis aus Leistung, Verständlichkeit und Praxistauglichkeit.

---

### 10. K – Knowledge Transfer: Ergebnisse nutzbar machen

In der K-Phase geht es darum, die Ergebnisse aus dem Projekt in eine verständliche und nutzbare Form zu bringen.

Mögliche Ergebnisse:

* dokumentiertes Jupyter Notebook
* GitHub Repository
* Streamlit Web-App
* Projektbericht
* Portfolio-Eintrag
* Präsentation für Stakeholder

Im akademischen Kontext liegt der Fokus auf sauberer Dokumentation.

In der Praxis ist zusätzlich wichtig, dass ein Modell produktiv genutzt werden kann.

> [!important]
> Ein ML-Projekt ist erst dann wirklich wertvoll, wenn die Ergebnisse verständlich kommuniziert und praktisch nutzbar gemacht werden.

---

### 11. Von Analyse zu Produktion

Der moderne Ansatz verbindet QUA³CK mit MLOps.

| QUA³CK-Phase | Klassischer Ansatz    | Moderner MLOps-Ansatz           |
| ------------ | --------------------- | ------------------------------- |
| Q + U        | statische Notebooks   | interaktive Analyse-Apps        |
| A³           | lokale Experimente    | MLFlow Experiment Tracking      |
| C            | manuelle Reports      | automatisierter Modellvergleich |
| K            | lokale Bereitstellung | GitHub + Streamlit Cloud        |

Dadurch entsteht eine Verbindung zwischen:

```text

Fragestellung → Datenverständnis → Modellierung → Bewertung → Deployment

```

---

### 12. Portfolio-Relevanz

Für das Portfolio sollte ein Projekt nicht nur Code enthalten, sondern den gesamten Prozess sichtbar machen.

Eine gute Portfolio-Struktur zeigt:

| Abschnitt  | Inhalt                                         |
| ---------- | ---------------------------------------------- |
| Problem    | Was soll gelöst werden?                        |
| Daten      | Welche Daten wurden verwendet?                 |
| EDA        | Welche Muster wurden erkannt?                  |
| Modelle    | Welche Algorithmen wurden getestet?            |
| Evaluation | Welches Modell war warum am besten?            |
| Deployment | Wie kann die Lösung genutzt werden?            |
| Reflexion  | Was wurde gelernt? Was wären nächste Schritte? |

Beispiel für das Iris-Projekt:

| Aspekt             | Inhalt                                      |
| ------------------ | ------------------------------------------- |
| Projekt            | AMALEA QUA³CK Demo – Iris Classification    |
| Methodik           | QUA³CK-Prozessmodell                        |
| Bester Algorithmus | Decision Tree                               |
| Performance        | 97,8 % Accuracy                             |
| Technologien       | Python, Pandas, Scikit-learn, Matplotlib    |
| Nächste Schritte   | MLFlow-Integration und Streamlit-Deployment |

---

### 13. Zentrale Begriffe

| Begriff                 | Kurzdefinition                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------- |
| **EDA**                 | Explorative Datenanalyse zur Untersuchung von Struktur, Qualität und Mustern in Daten |
| **Feature**             | Eingabevariable, die das Modell für Vorhersagen nutzt                                 |
| **Label**               | Zielvariable, die vorhergesagt werden soll                                            |
| **Hyperparameter**      | Einstellungen, die vor dem Training festgelegt werden                                 |
| **Overfitting**         | Modell lernt Trainingsdaten zu genau und generalisiert schlecht                       |
| **Underfitting**        | Modell ist zu einfach und erkennt Muster nicht ausreichend                            |
| **Train/Test-Split**    | Aufteilung in Trainingsdaten und unabhängige Testdaten                                |
| **Experiment Tracking** | systematische Dokumentation von Modellläufen                                          |
| **Deployment**          | Bereitstellung eines Modells zur praktischen Nutzung                                  |
| **Reproduzierbarkeit**  | Fähigkeit, Ergebnisse mit gleichen Daten und gleichem Code erneut zu erzeugen         |

---

### 14. Merksätze

> [!quote]
> QUA³CK ist ein strukturierter Weg von der Fragestellung bis zur Anwendung.

> [!quote]
> Die Q-Phase entscheidet, ob das richtige Problem gelöst wird.

> [!quote]
> Die U-Phase verhindert, dass blind modelliert wird.

> [!quote]
> Die A³-Phase ist iterativ: auswählen, anpassen, optimieren.

> [!quote]
> Die C-Phase entscheidet nicht nur nach Accuracy, sondern nach Gesamtqualität.

> [!quote]
> Die K-Phase macht aus einem Experiment ein nutzbares Ergebnis.

---

### 15. Prüfungs- und Verständnisfragen

1. Warum scheitern viele Data-Science-Projekte trotz guter technischer Umsetzung?
2. Wofür steht QUA³CK?
3. Warum ist die Q-Phase besonders wichtig?
4. Was ist der Unterschied zwischen X/y-Split und Train/Test-Split?
5. Welche Rolle spielt EDA in der U-Phase?
6. Was bedeuten die drei A in A³?
7. Warum reicht Accuracy allein nicht immer zur Modellauswahl?
8. Warum kann ein Decision Tree trotz gleicher Accuracy besser geeignet sein als KNN?
9. Welche Vorteile bietet MLFlow im ML-Prozess?
10. Was passiert in der K-Phase?
11. Wie verbindet MLOps Analyse und Produktion?
12. Welche Elemente sollte ein gutes Data-Science-Portfolio enthalten?

---

### 16. Mini-Zusammenfassung

Das QUA³CK-Modell strukturiert Machine-Learning-Projekte in fünf Phasen:

**Question**, **Understanding the Data**, **A³**, **Conclude & Compare** und **Knowledge Transfer**.

Die Einheit zeigt am Iris-Projekt, wie aus einer klaren Fragestellung über EDA, Modelltraining und Evaluation eine nutzbare Anwendung entstehen kann.

MLOps ergänzt den Prozess durch Experiment Tracking, Reproduzierbarkeit und Deployment-Vorbereitung.

---
### Aufgabe

>[!important]
>Suchen Sie sich mehrere hochlastige Datensätze aus kaggle o.ä. welche miteinander korrelieren und entwerfe Notebooks für die einzelnen QUACK Phasen zu einer von dir entworfenen Forschungsfrage
>- Datensätze finden und Forschungsfrage kristallisieren
>- Github Repo erstellen mit Jupiter Notebook, README füllen
>- GH Copilot / Claude Code / Codex nutzen 
>- Inspiration von: https://github.com/noahrsn/Degrees-of-No-Return-App
>
>Quellen:
>- UCI Machine Learning Repository (archive.ics.uci.edu/ml/index.php)
>- Kaggle Datasets (kaggle.com/datasets)
>- Google Dataset Search (datasetsearch.research.google.com)
>- AWS Open Data Registry (registry.opendata.aws)
>- Data.gov
>- European Data Portal (data.europa.eu)
>- OpenML (openml.org)


![[Data Analytics und Big Data_Datenquellen.png]]

## Einheit 2

...
