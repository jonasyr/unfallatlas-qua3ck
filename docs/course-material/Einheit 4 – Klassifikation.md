---
title: Einheit 4 – Klassifikation
description: Einheit 4
date: 12-05-2026
time: 12:35
reference: Data Analytics und Big Data
index: ""
subindex: ""
status:
  - begin
---

# Einheit 4 – Klassifikation

>- **Reference Link:** [[Data Analytics und Big Data]]

---
>[!summary]
> Diese Einheit ist die A³-Phase (Algorithm Selection) im Kleinformat: Am MNIST-Datensatz wird ein Binärklassifikator trainiert, mit Kreuzvalidierung bewertet und mit einer Reihe von Qualitätsmaßen seziert, die weit über die reine Accuracy hinausgehen.
>
> Der rote Faden: Accuracy allein ist ein Trap, sobald Klassen unbalanciert sind – ein Punkt, der in [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]] §9 ("Conclude & Compare") bereits angerissen wurde und hier mit Konfusionsmatrix, Präzision, Sensitivität, F1-Score, ROC-Kurve und AUC endlich mit Substanz gefüllt wird.

---

### 1. Warum braucht man mehr als eine Accuracy-Zahl?

Klassifikation – die Vorhersage diskreter Kategorien statt numerischer Werte – gilt neben der Regression als die häufigste Aufgabe des überwachten Lernens. Einen Klassifikator auszuwerten, ist dabei oft deutlich verzwickter als einen Regressor auszuwerten: Eine einzelne Prozentzahl "richtig klassifiziert" verschleiert leicht, welche Fehler ein Modell macht und wie teuer diese Fehler in der Praxis sind.

>[!note]
> Diese Einheit arbeitet durchgehend mit dem MNIST-Datensatz – 70.000 kleine Bilder handschriftlicher Ziffern, die als "Hello World" des Machine Learning gelten. Jedes neue Klassifikationsverfahren wird früher oder später an MNIST gemessen.

---

### 2. Einordnung in das QUA³CK-Modell

| Phase  | Bedeutung                                                         | Rolle in dieser Einheit                                                                 |
| ------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Q**  | Question                                                          | nicht direkt betroffen                                                                   |
| **U**  | Understanding the Data                                            | MNIST muss vor dem Training verstanden werden (Bildgröße, Pixelintensitäten, Klassenverteilung) |
| **A³** | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | Kernstück dieser Einheit: SGD, Random Forest, SVM, KNN werden ausgewählt, verglichen und bewertet |
| **C**  | Conclude & Compare                                                | dieselben Metriken wie in [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]] §9 (Accuracy, Precision, Recall, F1) – hier aber mit der Formel und dem Warum dahinter |
| **K**  | Knowledge Transfer                                                | nicht direkt betroffen                                                                   |

>[!note]
> Einheit 1 §9 listet Accuracy, Precision, Recall und F1-Score als quantitative Kriterien der C-Phase auf, ohne im Detail zu erklären, wie sie berechnet werden oder wann welche Metrik überhaupt sinnvoll ist. Genau diese Lücke füllt diese Einheit.

---

### 3. Lernziele der Einheit

Nach dieser Einheit solltest du:

- den MNIST-Datensatz und seine Rolle als ML-"Hello World" einordnen können
- einen Binärklassifikator trainieren und mit Kreuzvalidierung bewerten können
- Konfusionsmatrix, Präzision, Sensitivität und F1-Score berechnen und interpretieren können
- den Precision/Recall-Trade-off und die ROC-Kurve erklären können
- Multiklassen-, Multilabel- und Multioutput-Klassifikation unterscheiden können
- eine einfache Fehleranalyse anhand der Konfusionsmatrix durchführen können

>[!note]
> Zu dieser Einheit gehören außerdem ein Erklärvideo (*Wie_gut_ist_Ihre_KI_.mp4*), ein Podcast (*Warum_95_Prozent_Genauigkeit_wertlos_sind.m4a* — thematisch vermutlich der Precision/Recall-Trade-off) sowie zwei Foliensätze (*Klassifikation_Die_Suche_nach_der_Wahrheit_in_Daten.pdf*, *ML-Klassifikation.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert – bei Bedarf manuell sichten.

---

### 4. MNIST: der Datensatz

MNIST besteht aus 70.000 Bildern handschriftlicher Ziffern (0–9), gesammelt von US-Oberschülern und Mitarbeitern des Census Bureau. Jedes Bild hat 28 × 28 Pixel, also 784 Merkmale – eines pro Pixel, mit Werten von 0 (Weiß) bis 255 (Schwarz).

```python
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target.astype(np.uint8)  # Labels als Integer

# Datensatz kommt bereits vorgemischt und vorgeteilt:
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

>[!important]
> Der Testdatensatz wird beiseitegelegt, bevor die Daten genauer betrachtet werden – exakt wie schon in [[Einheit 2 – Understanding the Data - Datenexploration und Datenvorbereitung]] gefordert. Dass die Trainingsdaten bereits gemischt sind, ist kein Zufall: Bei der Kreuzvalidierung sollen alle Folds eine ähnliche Klassenverteilung haben, sonst fehlen in manchen Folds ganze Ziffern.

---

### 5. Trainieren eines Binärklassifikators

Um die Aufgabe zu vereinfachen, wird zunächst nur ein Binärklassifikator gebaut: "ist das eine 5 oder nicht". Ein solider Ausgangspunkt ist der `SGDClassifier` (stochastisches Gradientenverfahren), der Trainingsdatenpunkte einzeln verarbeitet und deshalb auch für sehr große Datensätze und Onlinelernen geeignet ist.

```python
from sklearn.linear_model import SGDClassifier

y_train_5 = (y_train == 5)  # True bei allen 5en, False bei allen anderen
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([X_train[0]])  # array([True]) – korrekt, X_train[0] ist eine 5
```

---

### 6. Leistungsmessung mit Kreuzvalidierung

Kreuzvalidierung liefert auch hier die üblichen Genauigkeitswerte, die auf den ersten Blick beeindruckend wirken:

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.95035, 0.96035, 0.9604])
```

Über 95% Accuracy auf allen Folds – das klingt hervorragend. Der Vergleich mit einer naiven Baseline entlarvt aber, wie wenig das bedeutet: Ein `DummyClassifier`, der stur "keine 5" tippt, kommt auf über 90%, einfach weil nur rund 10% aller Ziffern tatsächlich 5en sind.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.90965, 0.90965, 0.90965])
```

>[!important]
> Warum ist Accuracy bei stark unbalancierten Klassen (z. B. 90% Nicht-5en) irreführend? Weil ein Klassifikator, der einfach immer die Mehrheitsklasse vorhersagt, allein durch die Klassenverteilung eine hohe Accuracy erreicht – ohne irgendetwas gelernt zu haben. Die naive Baseline schlägt dann fast die naiven Erwartungen an das "richtige" Modell. Accuracy ist deshalb bei unbalancierten Datensätzen für gewöhnlich nicht das Qualitätsmaß der Wahl.

---

### 7. Die Konfusionsmatrix

Die Konfusionsmatrix zählt, wie oft Instanzen einer Kategorie A als Kategorie B vorhergesagt wurden. Für den Binärfall ("5" vs. "nicht-5") ergibt sich folgendes Schema:

| | Vorhergesagt: negativ | Vorhergesagt: positiv |
| --- | --- | --- |
| **Tatsächlich: negativ** | Richtig Negativ (RN) | Falsch Positiv (FP) |
| **Tatsächlich: positiv** | Falsch Negativ (FN) | Richtig Positiv (RP) |

Berechnet wird sie aus sauberen, "out of sample" gewonnenen Vorhersagen via `cross_val_predict()` (statt der Testdaten, die weiterhin unangetastet bleiben):

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
# array([[53892,   687],
#        [ 1891,  3530]])

ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred)
```

Ein perfekter Klassifikator hätte ausschließlich Werte ungleich null auf der Hauptdiagonale (RN und RP), alle Nebendiagonalfelder (FP, FN) wären leer.

---

### 8. Präzision und Sensitivität

Aus der Konfusionsmatrix lassen sich zwei kompaktere Maße ableiten:

**Präzision (Precision, "Relevanz")** – wie viele der als positiv vorhergesagten Fälle waren tatsächlich positiv?

$$\text{Präzision} = \frac{RP}{RP + FP}$$

**Sensitivität (Recall, "Trefferquote")** – wie viele der tatsächlich positiven Fälle wurden gefunden?

$$\text{Sensitivität} = \frac{RP}{RP + FN}$$

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred)  # 0.8370...
recall_score(y_train_5, y_train_pred)     # 0.6511...
```

Damit glänzt der 5-Detektor deutlich weniger, als es die Accuracy vermuten ließ: Nur 83,7% seiner positiven Vorhersagen sind korrekt, und er findet nur 65,1% aller tatsächlichen 5en.

Wer eine einzelne Kennzahl braucht, um zwei Klassifikatoren zu vergleichen, kombiniert Präzision und Sensitivität zum **F1-Score**, dem harmonischen Mittel beider Größen. Der harmonische Mittelwert gewichtet niedrige Werte stärker als der gewöhnliche Durchschnitt – ein hoher F1-Score ist also nur erreichbar, wenn *beide* Größen hoch sind:

```python
f1_score(y_train_5, y_train_pred)  # 0.7325...
```

>[!important]
> Eine perfekte Präzision lässt sich trivial erreichen, indem ein Klassifikator nur bei der einen Instanz, bei der er sich am sichersten ist, positiv vorhersagt und sonst immer negativ – Präzision 100%, aber praktisch nutzlos, weil die Sensitivität gegen null geht. Präzision ohne Sensitivität ist eine Kennzahl ohne Substanz.

---

### 9. Der Präzision/Sensitivität-Kompromiss

Der SGDClassifier berechnet für jede Instanz einen Score über eine Entscheidungsfunktion; liegt der Score über einem Schwellenwert, wird positiv klassifiziert. Je höher dieser Schwellenwert liegt, desto höher (im Allgemeinen) die Präzision, aber desto niedriger die Sensitivität – und umgekehrt. Beide Größen lassen sich also nicht gleichzeitig maximieren.

```python
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                              method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

Welche der beiden Größen wichtiger ist, hängt vollständig vom Anwendungsfall ab:

| Anwendungsfall | Priorität | Begründung |
| --- | --- | --- |
| Spamfilter / Jugendschutz-Filter | hohe Präzision | lieber ein paar gute Mails/Videos verwerfen, als etwas Ungeeignetes durchzulassen |
| Überwachungssystem (z. B. Ladendiebstahl erkennen) | hohe Sensitivität | ein paar Fehlalarme sind akzeptabel, solange möglichst kein echter Vorfall übersehen wird |

>[!note]
> Merkregel aus dem Buch: Wenn jemand sagt "Lass uns 99% Präzision erreichen", lautet die Gegenfrage immer: "Bei welcher Sensitivität?" Eine Präzisionsangabe ohne Sensitivitätsangabe ist wertlos.

---

### 10. Die ROC-Kurve

Die ROC-Kurve (Receiver Operating Characteristic) ist der Precision/Recall-Kurve eng verwandt, trägt aber die Richtig-positiv-Rate (TPR, identisch mit Sensitivität) gegen die Falsch-positiv-Rate (FPR) auf – den Anteil negativer Datenpunkte, die fälschlich als positiv eingestuft werden.

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_auc_score(y_train_5, y_scores)  # 0.9604... – 1.0 wäre perfekt, 0.5 wäre Zufall
```

Klassifikatoren lassen sich über die **AUC** (Area under the Curve) vergleichen: 1,0 für einen perfekten, 0,5 für einen rein zufälligen Klassifikator.

| Kurve | Wann bevorzugen? |
| --- | --- |
| Precision/Recall-Kurve | positive Klasse ist selten, oder falsch Positive sind teurer als falsch Negative |
| ROC-Kurve | ausgewogenere Klassenverteilung, kein starker Fokus auf falsch Positive |

>[!important]
> Bei stark unbalancierten Klassen (wie 5 vs. nicht-5) kann die ROC-AUC täuschend gut aussehen, weil es schlicht wenige Positive im Vergleich zu den Negativen gibt. Die PR-Kurve zeigt in solchen Fällen ehrlicher, wie viel Luft nach oben noch bleibt.

---

### 11. Multiklassenklassifikation

Multiklassenklassifikatoren (multinomiale Klassifikatoren) unterscheiden mehr als zwei Kategorien. Manche Algorithmen (`LogisticRegression`, `RandomForestClassifier`) können das direkt; andere (`SGDClassifier`, `SVC`) sind grundsätzlich binär und benötigen eine Strategie:

- **One-versus-Rest (OvR/OvA)**: ein Binärklassifikator pro Kategorie (z. B. 10 für MNIST), Vorhersage = Kategorie mit dem höchsten Score.
- **One-versus-One (OvO)**: ein Binärklassifikator pro Kategorienpaar (N × (N−1) / 2, für MNIST also 45), Vorhersage = Kategorie, die die meisten Duelle gewinnt. Vorteil: jeder Klassifikator trainiert nur auf den Daten der zwei relevanten Kategorien – praktisch bei Algorithmen wie SVM, die schlecht mit der Datensatzgröße skalieren.

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # Scikit-Learn wählt hier automatisch OvO

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
len(ovr_clf.estimators_)  # 10
```

Scikit-Learn wählt automatisch OvO oder OvR, je nach Algorithmus – SVC bevorzugt OvO wegen schlechter Skalierung, die meisten anderen binären Algorithmen (etwa SGDClassifier) laufen unter der Haube mit OvR.

---

### 12. Fehleranalyse

Bei zehn statt zwei Kategorien wird die Konfusionsmatrix schnell unübersichtlich – ein farbiges Diagramm hilft:

```python
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred, normalize="true", values_format=".0%"
)
```

Das Normalisieren nach Zeilen (`normalize="true"`) zeigt, welcher Anteil jeder tatsächlichen Kategorie korrekt bzw. falsch klassifiziert wurde – wichtig, weil sonst Kategorien mit vielen Beispielen die Matrix optisch dominieren, ohne dass das etwas über die Modellqualität aussagt. Setzt man zusätzlich `sample_weight` auf die falsch klassifizierten Fälle, treten die tatsächlichen Verwechslungsmuster (z. B. "viele Ziffern werden fälschlich als 8 erkannt") viel deutlicher hervor.

>[!note]
> Konfusionsmatrizen sind grundsätzlich nicht symmetrisch: dass 10% aller 5en als 8 fehlklassifiziert werden, sagt nichts darüber aus, wie viele 8en fälschlich als 5 erkannt werden (im Buch sind es nur 2%).

Aus so einer Analyse lassen sich konkrete Verbesserungsansätze ableiten: mehr Trainingsdaten für die häufig verwechselten Ziffern sammeln, neue Merkmale konstruieren (z. B. Anzahl geschlossener Schleifen: eine 8 hat zwei, eine 6 eine, eine 5 keine), oder die Bilder vorverarbeiten, um bekannte Verwechslungsmuster wie Verschiebungen und Rotationen zu entschärfen.

---

### 13. Multilabel-Klassifikation

Manchmal soll ein Klassifikator mehrere Labels gleichzeitig für dieselbe Instanz ausgeben – etwa bei Gesichtserkennung ("Alice: ja, Bob: nein, Charlie: ja"). Am MNIST-Beispiel lässt sich das mit zwei künstlichen Labels demonstrieren: "ist die Ziffer groß (≥7)?" und "ist die Ziffer ungerade?".

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train.astype("int8") % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

Zur Auswertung lässt sich der F1-Score pro Label berechnen und anschließend mitteln. Der `average`-Parameter steuert dabei, wie die Mittelung erfolgt:

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")     # alle Labels gleich gewichtet
f1_score(y_multilabel, y_train_knn_pred, average="weighted")  # nach Support gewichtet
```

Klassifikatoren, die keine Multilabel-Vorhersage direkt unterstützen (z. B. `SVC`), lassen sich über `ClassifierChain` verketten – jedes Modell in der Kette nutzt zusätzlich zu den Merkmalen die Vorhersagen der vorherigen Modelle, wodurch Abhängigkeiten zwischen Labels erfasst werden können (z. B. dass eine große Ziffer mit höherer Wahrscheinlichkeit auch ungerade ist).

---

### 14. Multioutput-Klassifikation

Die Verallgemeinerung der Multilabel-Klassifikation: Jedes Label kann nun selbst mehr als zwei mögliche Werte annehmen. Klassisches Beispiel: Rauschen aus Bildern entfernen – Eingabe ist ein verrauschtes Ziffernbild, Ausgabe ein Array von Pixelintensitäten (ein "Label" pro Pixel, mit Werten 0–255 statt nur True/False).

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

# Denoising-Beispiel:
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train  # Ziel: die sauberen Originalbilder
y_test_mod = X_test

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
```

>[!note]
> Die Grenze zwischen Klassifikation und Regression verschwimmt hier bewusst: Die Vorhersage von Pixelintensitäten ließe sich ebenso gut als Regressionsaufgabe auffassen. Multioutput-Systeme sind zudem nicht auf Klassifikation beschränkt – auch Mischformen aus Kategorien und numerischen Werten pro Datenpunkt sind denkbar.

---

### 15. Praktische Übung zur Einheit

| Aufgabe | Inhalt |
| --- | --- |
| 1. Hyperparametersuche | Finde per Gittersuche (`weights`, `n_neighbors`) Hyperparameter für einen `KNeighborsClassifier`, mit denen du auf den MNIST-Testdaten über 97% Accuracy erreichst. |
| 2. Data Augmentation | Schreibe eine Funktion, die ein MNIST-Bild um ein Pixel in jede Richtung verschiebt. Erweitere den Trainingsdatensatz um die verschobenen Kopien und miss, ob sich die Testaccuracy deines besten Modells verbessert. |
| 3. Titanic-Klassifikator | Trainiere auf dem Titanic-Datensatz (Kaggle oder `https://homl.info/titanic.tgz`) einen Klassifikator, der die Spalte `Survived` aus den übrigen Spalten vorhersagt. |
| 4. Spamfilter-Pipeline | Baue aus dem Apache-SpamAssassin-Datensatz eine Vorverarbeitungspipeline (E-Mail → Merkmalsvektor über Wortvorkommen), trainiere mehrere Klassifikatoren und optimiere auf hohe Präzision *und* Sensitivität gleichzeitig. |

---

### 16. Zentrale Begriffe

| Begriff | Kurzdefinition |
| --- | --- |
| **Konfusionsmatrix** | Tabelle, die zählt, wie oft Instanzen einer tatsächlichen Kategorie als welche vorhergesagte Kategorie klassifiziert wurden |
| **Präzision (Precision)** | Anteil der als positiv vorhergesagten Fälle, die tatsächlich positiv sind: RP / (RP + FP) |
| **Sensitivität (Recall)** | Anteil der tatsächlich positiven Fälle, die korrekt erkannt wurden: RP / (RP + FN) |
| **F1-Score** | harmonisches Mittel aus Präzision und Sensitivität, hoch nur wenn beide hoch sind |
| **ROC-Kurve** | trägt Richtig-positiv-Rate gegen Falsch-positiv-Rate für alle Schwellenwerte auf |
| **AUC** | Fläche unter der ROC-Kurve; 1,0 = perfekt, 0,5 = Zufallsklassifikator |
| **Multiklassenklassifikation** | Unterscheidung von mehr als zwei Kategorien, direkt oder über OvR/OvO |
| **Multilabel-Klassifikation** | ein Klassifikator gibt mehrere (binäre) Labels pro Instanz aus |
| **Multioutput-Klassifikation** | Verallgemeinerung von Multilabel, bei der jedes Label mehr als zwei Werte annehmen kann |
| **Baseline-Klassifikator** | naiver Vergleichsklassifikator (z. B. `DummyClassifier`), der zeigt, wie viel ein Modell wirklich leistet |
| **Data Augmentation** | künstliche Vergrößerung des Trainingsdatensatzes durch leicht veränderte Kopien vorhandener Beispiele |

---

### 17. Merksätze

>[!quote]
> Eine hohe Accuracy bei unbalancierten Klassen ist keine Leistung – sie ist Arithmetik.

>[!quote]
> Präzision ohne Sensitivität ist eine Kennzahl ohne Substanz, und umgekehrt.

>[!quote]
> Wer "99% Präzision" verspricht, ohne die Sensitivität zu nennen, verspricht nicht viel.

>[!quote]
> Die Konfusionsmatrix lügt nicht – aber sie ist auch nicht symmetrisch.

>[!quote]
> Ein Baseline-Klassifikator ist kein Strohmann, sondern der Maßstab, an dem sich jedes "richtige" Modell zuerst blamieren muss.

---

### 18. Prüfungs- und Verständnisfragen

1. Warum ist Accuracy bei stark unbalancierten Klassen (z. B. 90% Nicht-5en) irreführend?
2. Was zeigt der Vergleich zwischen SGDClassifier und `DummyClassifier` in dieser Einheit über den Wert von Baseline-Modellen?
3. Wie ist eine Konfusionsmatrix aufgebaut, und was bedeuten RP, FP, FN und RN konkret?
4. Wie werden Präzision und Sensitivität berechnet, und warum lässt sich eine perfekte Präzision trivial, aber nutzlos erreichen?
5. Was leistet der F1-Score, den Präzision und Sensitivität einzeln nicht leisten?
6. Nenne ein Beispiel, bei dem hohe Präzision wichtiger ist als hohe Sensitivität, und eines, bei dem es umgekehrt ist.
7. Wann bevorzugt man die ROC-Kurve, wann die Precision/Recall-Kurve?
8. Was unterscheidet die One-versus-Rest- von der One-versus-One-Strategie bei Multiklassenklassifikation?
9. Wie hilft das Normalisieren der Konfusionsmatrix nach Zeilen bei der Fehleranalyse?
10. Was ist der Unterschied zwischen Multilabel- und Multioutput-Klassifikation?
11. Entwickle einen Klassifikator für den MNIST-Datensatz, der auf den Testdaten eine Genauigkeit von mehr als 97% erzielt (Hinweis: `KNeighborsClassifier` mit passenden Hyperparametern für `weights` und `n_neighbors`).
12. Schreibe eine Funktion, die ein MNIST-Bild pixelweise in jede Richtung verschiebt, erweitere damit den Trainingsdatensatz und miss die Wirkung auf die Testaccuracy (Data Augmentation).
13. Trainiere einen Klassifikator auf dem Titanic-Datensatz, der die Spalte `Survived` vorhersagt.
14. Entwirf eine Vorverarbeitungspipeline und trainiere mehrere Klassifikatoren für einen Spamfilter auf Basis des SpamAssassin-Datensatzes.

---

### 19. Mini-Zusammenfassung

Ein Klassifikator lässt sich nicht an einer einzigen Zahl ablesen. Accuracy versagt bei unbalancierten Klassen sichtbar, wie der Vergleich mit einem naiven `DummyClassifier` zeigt. Die Konfusionsmatrix liefert die Rohdaten (RP, FP, FN, RN), aus denen sich Präzision, Sensitivität und F1-Score ableiten lassen – wobei Präzision und Sensitivität in einem unvermeidlichen Trade-off zueinander stehen, den die Precision/Recall-Kurve und die ROC-Kurve (mit ihrer AUC) sichtbar machen. Über die binäre Klassifikation hinaus lassen sich mit OvR/OvO-Strategien Multiklassenprobleme lösen, mit Multilabel-Systemen mehrere Labels pro Instanz und mit Multioutput-Systemen sogar mehrwertige Labels vorhersagen. Fehleranalyse anhand der (normalisierten) Konfusionsmatrix zeigt schließlich systematisch, wo ein Modell noch schwächelt und wo sich weitere Arbeit lohnt.

---

### Aufgabe

>[!important]
> Wende auf den Klassifikator deines eigenen QUA³CK-Projekts (oder ein geplantes Projekt) eine vollständige Fehleranalyse an:
>- Berechne die Konfusionsmatrix für dein Modell und interpretiere sie (normalisiert nach Zeilen und, falls sinnvoll, nach Spalten).
>- Berechne Präzision, Sensitivität und F1-Score und begründe, welche der beiden Größen (Präzision oder Sensitivität) für dein Projekt wichtiger ist – und warum.
>- Beschreibe mindestens einen konkreten Verbesserungsansatz, den deine Fehleranalyse nahelegt (z. B. zusätzliche Trainingsdaten, neue Merkmale, Data Augmentation).
