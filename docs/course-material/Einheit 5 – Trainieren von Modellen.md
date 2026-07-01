---
title: Einheit 5 – Trainieren von Modellen
description: Einheit 5
date: 12-05-2026
time: 12:45
reference: Data Analytics und Big Data
index: ""
subindex: ""
status:
  - begin
---

# Einheit 5 – Trainieren von Modellen

>- **Reference Link:** [[Data Analytics und Big Data]]

---
>[!summary]
> [[Einheit 3 – Die Machine-Learning-Umgebung]] und [[Einheit 4 – Klassifikation]] haben Modelle bislang als Black Box behandelt: Man ruft `.fit()` und `.predict()` auf und bewertet das Ergebnis, ohne zu wissen, was dazwischen passiert. Diese Einheit öffnet genau diese Box – am Beispiel der linearen Regression.
>
> Der rote Faden: Ein Modell zu trainieren heißt, eine Kostenfunktion zu minimieren – entweder in einem Schritt über eine geschlossene Gleichung (Normalengleichung) oder iterativ über ein Gradientenverfahren. Mit mehr Modellkomplexität (polynomielle Regression) taucht das Overfitting-Problem aus [[Einheit 3 – Die Machine-Learning-Umgebung]] §13 in voller Härte wieder auf – und wird hier mit Lernkurven diagnostiziert und mit Regularisierung (Ridge, Lasso, Elastic Net, Early Stopping) behandelt. Zum Schluss wird das gleiche Handwerkszeug auf Klassifikation angewendet: logistische und Softmax-Regression.

---

### 1. Die Black Box öffnen

Ein Modell zu trainieren bedeutet, seine Parameter so einzustellen, dass es die Trainingsdaten möglichst gut abbildet. Bisher genügte dazu ein Aufruf von `.fit()`. Diese Einheit fragt, was `.fit()` tatsächlich berechnet – am Beispiel des wohl einfachsten Modells überhaupt, der linearen Regression.

Zwei grundsätzlich verschiedene Trainingsstrategien stehen zur Wahl:

- eine **Gleichung mit geschlossener Form**, die die optimalen Parameter in einem Rechenschritt direkt liefert (Normalengleichung)
- ein **iteratives Optimierungsverfahren**, das Gradientenverfahren (GD), das die Parameter schrittweise anpasst, um eine Kostenfunktion zu minimieren

>[!note]
> Warum das lohnt: Ein Grundverständnis der Trainingsverfahren hilft dabei, schnell ein passendes Modell samt Trainingsverfahren und Hyperparametern zu finden, erleichtert die Fehlersuche – und ist die Grundlage für alles, was später zu neuronalen Netzen führt.

Von der linearen Regression aus führt der Weg über die polynomielle Regression (die auch nichtlineare Daten mit einem linearen Modell abbildet, aber anfälliger für Overfitting ist) zu Regularisierungstechniken, und schließlich zu zwei Modellen für Klassifikationsaufgaben: logistische Regression und Softmax-Regression.

---

### 2. Einordnung in das QUA³CK-Modell

| Phase  | Bedeutung                                                         | Rolle in dieser Einheit                                                                 |
| ------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Q**  | Question                                                          | nicht direkt betroffen                                                                   |
| **U**  | Understanding the Data                                            | Merkmalsskalierung ist Voraussetzung für ein funktionierendes Gradientenverfahren        |
| **A³** | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | tiefster Tauchgang der Reihe in **Adjusting Hyperparameters**: `eta` (Lernrate), `alpha` (Regularisierungsstärke), `degree` (Polynomgrad) werden hier erstmals mechanistisch erklärt statt nur per Gittersuche optimiert |
| **C**  | Conclude & Compare                                                | Lernkurven liefern ein zusätzliches Diagnosewerkzeug neben den Metriken aus [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]] §9 |
| **K**  | Knowledge Transfer                                                | nicht direkt betroffen                                                                   |

>[!note]
> Diese Einheit ist die mechanistische Fortsetzung von [[Einheit 3 – Die Machine-Learning-Umgebung]] §13: Dort wurden Overfitting und Underfitting nur als Symptome mit generischen Gegenmaßnahmen ("Modell vereinfachen", "Restriktionen lockern") beschrieben. Hier bekommen diese Gegenmaßnahmen konkrete Hyperparameter-Namen.

Die Hyperparameter dieser Einheit lassen sich direkt den Fehlerbildern aus [[Einheit 3 – Die Machine-Learning-Umgebung]] §13 zuordnen:

| Fehlerbild (Einheit 3 §13) | Hyperparameter dieser Einheit | Wirkrichtung |
| --- | --- | --- |
| Overfitting | `alpha` erhöhen (Ridge/Lasso/Elastic Net) | stärkere Regularisierung, kleinere Gewichte |
| Overfitting | Early Stopping aktivieren | Training endet, bevor der Validierungsfehler wieder steigt |
| Overfitting | `degree` bei `PolynomialFeatures` senken | weniger Modellkapazität |
| Underfitting | `degree` bei `PolynomialFeatures` erhöhen | mehr Modellkapazität |
| Underfitting | `alpha` senken | schwächere Regularisierung, mehr Freiheit für die Gewichte |
| Divergenz/lange Trainingsdauer | `eta` (Lernrate) anpassen | zu groß → Divergenz, zu klein → langsame Konvergenz |

---

### 3. Lernziele der Einheit

Nach dieser Einheit solltest du:

- die Normalengleichung und ihre Rechenkomplexität erklären können
- Batch-, Stochastisches und Mini-Batch-Gradientenverfahren unterscheiden können
- polynomielle Regression und Lernkurven zur Diagnose von Über-/Unteranpassung nutzen können
- Ridge-, Lasso- und Elastic-Net-Regularisierung unterscheiden und begründet auswählen können
- Early Stopping als Regularisierungstechnik einordnen können
- logistische und Softmax-Regression für Klassifikationsaufgaben anwenden können

>[!note]
> Zu dieser Einheit gehören außerdem ein Erklärvideo (*Training_von_ML-Modellen.mp4*), ein Podcast (*Unter_der_Haube_von_Regression_und_Gradientenverfahren.m4a*) sowie ein Foliensatz (*ML_Modelle_Black_Box_öffnen.pdf*) und ein zusätzliches 46-seitiges Slide-Deck (*ML-Modelltraining.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert – bei Bedarf manuell sichten.

---

### 4. Lineare Regression: Grundidee und Kostenfunktion

Ein lineares Modell trifft seine Vorhersage als gewichtete Summe der Eingabemerkmale plus einer Konstante, dem Bias-Term:

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

In Vektorschreibweise: $\hat{y} = h_\theta(x) = \theta \cdot x$, wobei $\theta$ der Parametervektor des Modells (inklusive Bias $\theta_0$) und $x$ der Merkmalsvektor eines Datenpunkts ist (mit $x_0 = 1$).

Um dieses Modell zu trainieren, braucht es ein Gütekriterium. In der Praxis nimmt man dafür statt des RMSE lieber den **mittleren quadratischen Fehler (MSE)** als Kostenfunktion – beide werden am selben Punkt minimal, aber der MSE lässt sich einfacher optimieren:

$$\text{MSE}(\theta) = \frac{1}{m}\sum_{i=1}^{m}\left(\theta^\top x^{(i)} - y^{(i)}\right)^2$$

Training heißt also: denjenigen Wert für $\theta$ finden, der diese Kostenfunktion über den Trainingsdaten minimiert.

>[!note]
> Trainings- und Bewertungsmaß müssen nicht identisch sein. Klassifikatoren werden häufig über eine leicht optimierbare Kostenfunktion (z. B. Log Loss, siehe §16) trainiert, aber über Precision/Recall bewertet – solange beide Größen stark korrelieren, ist das kein Widerspruch.

---

### 5. Die Normalengleichung

Für den MSE einer linearen Regression gibt es eine Lösung mit geschlossener Form, die $\theta$ direkt berechnet – die **Normalengleichung**:

$$\hat{\theta} = (X^\top X)^{-1} X^\top y$$

```python
import numpy as np
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)
m = 100  # Anzahl Instanzen
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = add_dummy_feature(X)  # Bias-Spalte (x0 = 1) hinzufügen

theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("Normalengleichung theta:", theta_best.ravel())
# Erwartetes Ergebnis: ca. [4, 3]
```

Die Daten wurden künstlich aus $y = 4 + 3x_1 + \text{Rauschen}$ generiert – die Normalengleichung findet daraus $\theta$-Werte nahe $[4, 3]$, aber nicht exakt, weil das Rauschen eine perfekte Rekonstruktion unmöglich macht.

>[!warning]
> **Rechenkomplexität**: Die Normalengleichung invertiert eine $(n+1) \times (n+1)$-Matrix, das kostet etwa $O(n^{2{,}4})$ bis $O(n^3)$ Rechenzeit (n = Anzahl Merkmale). Scikit-Learns SVD-basierter Ansatz (den `LinearRegression` tatsächlich verwendet) liegt bei $O(n^2)$ – besser, aber bei sehr vielen Merkmalen (z. B. 100.000) werden beide Verfahren unpraktikabel langsam. In der Anzahl der Trainingsinstanzen $m$ sind beide dagegen linear ($O(m)$) und verarbeiten daher große Datensätze problemlos, solange sie in den Speicher passen. Bei sehr vielen Merkmalen ist ein Gradientenverfahren die bessere Wahl.

---

### 6. Das Gradientenverfahren: Grundprinzip

Das Gradientenverfahren (Gradient Descent, GD) ist ein allgemeiner Optimierungsalgorithmus: Er verändert die Parameter iterativ in die Richtung, in der die Kostenfunktion am steilsten fällt – wie das Abstiegstempo im dichten Nebel am Berg, wo man nur die Neigung unter den Füßen spürt. Sobald der Gradient null wird, ist ein Minimum erreicht.

Der zentrale Hyperparameter ist die **Lernrate** `eta` ($\eta$): Sie legt die Schrittgröße fest.

>[!important]
> Ist die Lernrate zu klein, dauert die Konvergenz sehr lange. Ist sie zu groß, kann der Algorithmus über das Minimum hinausspringen und **divergieren** – die Kostenfunktion wird dann von Schritt zu Schritt größer statt kleiner.

Der MSE als Kostenfunktion der linearen Regression ist konvex und stetig differenzierbar – es gibt also nur ein globales Minimum, kein Risiko lokaler Minima. Das gilt nicht für jede Kostenfunktion: Bei unregelmäßigeren Landschaften kann eine zufällige Initialisierung in einem lokalen Minimum oder auf einem langen Plateau stranden.

>[!warning]
> Sind Merkmale unterschiedlich skaliert, wird die Kostenfunktion zu einer länglichen statt einer runden Schüssel, und das Gradientenverfahren braucht deutlich länger, um das Minimum zu erreichen. Merkmale vor dem Gradientenverfahren immer skalieren (z. B. mit `StandardScaler`).

---

### 7. Batch-Gradientenverfahren

Beim Batch-Gradientenverfahren wird bei jedem Schritt der Gradient über den **gesamten** Trainingsdatensatz berechnet:

$$\nabla_\theta \text{MSE}(\theta) = \frac{2}{m} X^\top (X\theta - y)$$

```python
# X_b, m, y aus §5 (Normalengleichung) übernommen
eta = 0.1
n_epochs = 1000

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print("Batch-GD theta:", theta.ravel())
```

Jeder vollständige Durchlauf durch die Trainingsdaten heißt **Epoche**. Mit einer passenden Lernrate liefert dieser Code (nach 1000 Epochen) exakt das gleiche Ergebnis wie die Normalengleichung in §5.

Der Nachteil: Bei jedem einzelnen Schritt wird der gesamte Datensatz durchgerechnet, was das Verfahren bei großen Trainingsdatensätzen langsam macht – dafür skaliert es gut mit der Anzahl der Merkmale.

---

### 8. Stochastisches Gradientenverfahren

Das stochastische Gradientenverfahren (SGD) geht ins andere Extrem: Es wählt bei jedem Schritt nur **einen zufälligen Datenpunkt** und berechnet den Gradienten nur für diesen. Das macht jeden Schritt viel schneller und ermöglicht Training auf riesigen Datensätzen (auch Out-of-Core), aber der Weg zum Minimum wird viel unregelmäßiger – die Kostenfunktion hüpft, statt sanft zu sinken, und kommt am Minimum nie ganz zur Ruhe.

Die übliche Lösung: die Lernrate schrittweise über einen **Learning Schedule** senken (Simulated Annealing) – große Schritte am Anfang, kleine gegen Ende.

```python
# X_b, m, y aus §5 übernommen
t0, t1 = 5, 50  # Hyperparameter für den Learning Schedule

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)
n_epochs = 50

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)  # für SGD nicht durch m teilen
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

print("SGD theta:", theta.ravel())
```

Mit nur 50 statt 1000 Epochen erreicht SGD bereits eine gute Lösung. Die Zufälligkeit hilft zudem, unregelmäßige Kostenfunktionen zu verlassen, wenn sie in einem lokalen Minimum feststecken – ein Vorteil gegenüber dem Batch-Verfahren, sobald Kostenfunktionen nicht mehr konvex sind.

>[!note]
> Damit SGD im Mittel wirklich Richtung globalem Optimum wandert, müssen die Trainingsinstanzen unabhängig und identisch verteilt sein (z. B. durch Durchmischen). Sind sie nach Label sortiert, optimiert SGD nacheinander pro Label und verfehlt das globale Minimum.

---

### 9. Mini-Batch-Gradientenverfahren und SGD mit scikit-learn

Das Mini-Batch-Gradientenverfahren liegt zwischen beiden Extremen: Es berechnet Gradienten auf kleinen, zufälligen Teilmengen (Mini-Batches) statt auf dem gesamten Datensatz oder nur einem Punkt. Der Hauptvorteil gegenüber SGD: für Matrizenoperationen optimierte Hardware (insbesondere GPUs) lässt sich ausnutzen.

In der Praxis übernimmt `SGDRegressor` aus scikit-learn diese Arbeit:

```python
# X, y aus §5 übernommen
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None,
                       eta0=0.01, n_iter_no_change=100,
                       random_state=42)
sgd_reg.fit(X, y.ravel())
print("SGDRegressor intercept:", sgd_reg.intercept_, "coef:", sgd_reg.coef_)
```

`SGDRegressor` trainiert entweder 1000 Epochen (`max_iter`) oder stoppt früher, wenn sich der Verlust über 100 Epochen (`n_iter_no_change`) um weniger als `tol` verbessert.

| Verfahren | Konvergenzverhalten | Rechenaufwand pro Schritt | Eignung für große Datensätze |
| --- | --- | --- | --- |
| Batch-GD | glatt, endet exakt im Minimum | hoch (ganzer Datensatz) | schlecht (langsam bei großem $m$), gut bei großem $n$ |
| Stochastisches GD | unregelmäßig, pendelt um das Minimum | sehr niedrig (1 Datenpunkt) | sehr gut, Out-of-Core-fähig |
| Mini-Batch-GD | ruhiger als SGD, pendelt enger um das Minimum | niedrig (kleine Teilmenge) | gut, zusätzlich GPU-freundlich |

>[!tip]
> Nach dem Training gibt es kaum noch Unterschiede zwischen den Verfahren: Alle drei liefern sehr ähnliche Modelle. Die Wahl ist daher primär eine Frage der Trainingsgeschwindigkeit und Datensatzgröße, nicht der späteren Modellqualität.

---

### 10. Polynomielle Regression

Auch nichtlineare Daten lassen sich mit einem linearen Modell fitten: Man fügt einfach Potenzen jedes Merkmals als zusätzliche Merkmale hinzu und trainiert ein lineares Modell auf dem erweiterten Merkmalssatz. Scikit-Learns `PolynomialFeatures` übernimmt diese Transformation:

```python
PolynomialFeatures(degree=90, include_bias=False)
```

Der Hyperparameter `degree` steuert dabei direkt die Modellkapazität: `PolynomialFeatures(degree=d)` erzeugt aus $n$ Merkmalen $\binom{n+d}{d}$ neue Merkmale (inklusive aller Kombinationen bei mehreren Ausgangsmerkmalen) – bei hohem `degree` und mehreren Merkmalen droht eine kombinatorische Explosion der Merkmalsanzahl.

>[!warning]
> Ein hoher Polynomgrad fittet die Trainingsdaten immer genauer, treibt das Modell aber direkt ins Overfitting. Ein zu niedriger Grad (z. B. `degree=1`, also die gewöhnliche lineare Regression) führt bei nichtlinearen Daten zu Underfitting. `degree` ist damit ein klassischer Overfitting/Underfitting-Hebel im Sinne von [[Einheit 3 – Die Machine-Learning-Umgebung]] §13.

---

### 11. Lernkurven

Um zu diagnostizieren, ob ein Modell over- oder underfittet, hilft neben Kreuzvalidierung ein zweites Werkzeug: **Lernkurven**. Sie tragen Trainings- und Validierungsfehler über der Größe des Trainingsdatensatzes auf.

```python
# X, y aus §5 übernommen
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)
print("Lernkurve – letzter Trainingsfehler:", train_errors[-1].round(4),
      "| Validierungsfehler:", valid_errors[-1].round(4))
```

Die Kurvenform verrät direkt, welches Problem aus [[Einheit 3 – Die Machine-Learning-Umgebung]] §13 vorliegt:

| Kurvenbild | Diagnose | Typische Ursache |
| --- | --- | --- |
| beide Kurven laufen auf ein Plateau zu, liegen dicht beieinander und **hoch** | Underfitting | Modell zu einfach für die Datenstruktur |
| Trainingsfehler bleibt **niedrig**, Validierungsfehler bleibt deutlich darüber, große Lücke zwischen den Kurven | Overfitting | Modell zu komplex (z. B. hoher Polynomgrad) relativ zur Datenmenge |

>[!tip]
> Bei Underfitting hilft mehr Trainingsdaten nichts – hier braucht es ein mächtigeres Modell oder bessere Merkmale. Bei Overfitting kann mehr Trainingsdaten helfen, bis sich die Kurven annähern; alternativ Regularisierung (§12–§15) oder ein einfacheres Modell.

---

### 12. Regularisierte lineare Modelle: Ridge-Regression

Regularisierung schränkt die Freiheitsgrade eines Modells ein, um Overfitting zu erschweren – bei linearen Modellen normalerweise über eine Nebenbedingung auf den Gewichten. Die **Ridge-Regression** (Tikhonov-Regularisierung) addiert zum MSE einen Strafterm proportional zur quadrierten $\ell_2$-Norm des Gewichtsvektors:

$$J(\theta) = \text{MSE}(\theta) + \frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2$$

Der Bias-Term $\theta_0$ bleibt unangetastet (die Summe beginnt bei $i=1$). Der Hyperparameter `alpha` steuert die Stärke: bei $\alpha = 0$ entspricht Ridge der gewöhnlichen linearen Regression, bei sehr großem $\alpha$ werden alle Gewichte nahezu null.

```python
# X, y aus §5 übernommen
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

>[!warning]
> Ridge-Regression reagiert empfindlich auf die Skala der Eingabemerkmale – wie beim Gradientenverfahren gilt: **erst skalieren** (z. B. `StandardScaler`), **dann** regularisieren, nicht umgekehrt.

---

### 13. Lasso-Regression

Die **Lasso-Regression** (Least Absolute Shrinkage and Selection Operator) verwendet statt der $\ell_2$-Norm die $\ell_1$-Norm des Gewichtsvektors als Strafterm:

$$J(\theta) = \text{MSE}(\theta) + 2\alpha\sum_{i=1}^{n}|\theta_i|$$

```python
# X, y aus §5 übernommen
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```

Der entscheidende Unterschied zu Ridge: Lasso setzt die Gewichte unwichtiger Merkmale tendenziell vollständig auf null, statt sie nur zu verkleinern. Das ist gleichzeitig Nebeneffekt und Feature: Lasso betreibt damit implizit **automatische Merkmalsauswahl** und liefert ein sparsames Modell mit wenigen Gewichten ungleich null.

---

### 14. Elastic Net

**Elastic Net** mischt Ridge und Lasso über einen Mischparameter $r$ (`l1_ratio` in scikit-learn):

$$J(\theta) = \text{MSE}(\theta) + r \cdot 2\alpha\sum_{i=1}^{n}|\theta_i| + (1-r)\frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2$$

Bei $r=0$ entspricht Elastic Net reiner Ridge-Regression, bei $r=1$ reinem Lasso.

```python
# X, y aus §5 übernommen
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

Als Faustregel: Reine lineare Regression (ohne jede Regularisierung) sollte man meiden – ein wenig Regularisierung ist fast immer besser. Ridge ist ein solider Standard-Ausgangspunkt. Werden nur wenige Merkmale als wirklich relevant vermutet, sind Lasso oder Elastic Net vorzuziehen. Elastic Net wird dabei gegenüber reinem Lasso bevorzugt, weil Lasso instabil werden kann, wenn es mehr Merkmale als Trainingsinstanzen gibt oder Merkmale stark korrelieren.

---

### 15. Early Stopping

Ein völlig anderer Regularisierungsansatz für iterative Verfahren wie das Gradientenverfahren: das Training abbrechen, sobald der Validierungsfehler sein Minimum erreicht hat – **Early Stopping**. Trainiert man weiter, sinkt der Trainingsfehler zwar noch, aber der Validierungsfehler beginnt wieder zu steigen: das Modell overfittet.

```python
# X, y aus §5 übernommen
from copy import deepcopy
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

preprocessing = make_pipeline(
    PolynomialFeatures(degree=90, include_bias=False),  # wie in §10
    StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train.ravel())
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)

print(f"Early Stopping – bester Validierungsfehler: {best_valid_rmse:.4f}")
```

Statt `fit()` wird hier `partial_fit()` verwendet, um das Modell Epoche für Epoche inkrementell zu trainieren und zwischendurch auf dem Validierungsdatensatz zu messen.

>[!tip]
> Der Code stoppt das Training gar nicht wirklich ab – er merkt sich fortlaufend eine Kopie des bisher besten Modells via `deepcopy(sgd_reg)`. Das ist kein Zufall: Geoffrey Hinton nannte Early Stopping einen "beautiful free lunch", aber der Champagner gehört erst dem *besten* Modell, nicht dem *letzten*. Wer stattdessen `sgd_reg` selbst nach der Schleife verwendet, feiert möglicherweise ein bereits wieder overfittetes Modell.

---

### 16. Logistische Regression

Die **logistische Regression** schätzt die Wahrscheinlichkeit, dass ein Datenpunkt zu einer bestimmten Kategorie gehört, und ist damit trotz des Namens ein Klassifikationsverfahren. Statt wie die lineare Regression das Ergebnis der gewichteten Summe direkt auszugeben, wird es durch die logistische (sigmoide) Funktion geschickt:

$$\hat{p} = h_\theta(x) = \sigma(\theta^\top x), \qquad \sigma(t) = \frac{1}{1+\exp(-t)}$$

Das Ergebnis liegt zwischen 0 und 1. Trainiert wird über den **Log Loss** (Kreuzentropie im Binärfall) als Kostenfunktion; anders als bei der linearen Regression gibt es dafür keine geschlossene Lösung, aber die Funktion ist konvex, sodass ein Gradientenverfahren garantiert zum globalen Optimum konvergiert.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal width (cm)"]].values
y_iris = iris.target_names[iris.target] == 'virginica'

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_iris, y_iris, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_lr, y_train_lr)
print("Log. Regression – Wahrscheinlichkeit für petal_width=1.5:",
      log_reg.predict_proba([[1.5]]).round(3))
```

`predict_proba()` liefert die geschätzten Wahrscheinlichkeiten für beide Kategorien, `predict()` die daraus abgeleitete harte Vorhersage.

---

### 17. Entscheidungsgrenzen

Ist die geschätzte Wahrscheinlichkeit $\hat p \geq 0{,}5$, sagt das Modell die positive Kategorie vorher, sonst die negative – der Standard-Schwellenwert liegt also bei 50 %. Am Iris-Beispiel (Kronblattbreite) verlaufen beide Kategorien-Wahrscheinlichkeiten stetig gegenläufig; der Punkt, an dem sich beide Kurven bei 50 % schneiden, ist die **Entscheidungsgrenze** des Modells (im Buchbeispiel bei ca. 1,65 cm Kronblattbreite).

```python
# log_reg aus §16 übernommen
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]
```

Bei mehr als einem Merkmal wird aus der Entscheidungsgrenze eine Gerade (bzw. bei noch mehr Merkmalen eine Hyperebene) im Merkmalsraum. Zwei Kategorien mit überlappenden Merkmalsbereichen führen dabei zwangsläufig zu einem Unsicherheitsbereich um die Grenze herum, in dem das Modell sich nicht sicher ist, aber trotzdem eine Kategorie vorhersagen muss.

>[!note]
> Der Regularisierungs-Hyperparameter der logistischen Regression heißt in scikit-learn nicht `alpha`, sondern `C` – sein **Kehrwert**. Je größer `C`, desto schwächer die Regularisierung (umgekehrt zu `alpha` bei Ridge/Lasso/Elastic Net).

---

### 18. Softmax-Regression

Die **Softmax-Regression** (multinomiale logistische Regression) verallgemeinert logistische Regression direkt auf mehr als zwei Kategorien, ohne mehrere Binärklassifikatoren kombinieren zu müssen (im Gegensatz zu OvR/OvO aus [[Einheit 4 – Klassifikation]] §11). Für jede Kategorie $k$ wird ein eigener Score berechnet und anschließend über die Softmax-Funktion in eine Wahrscheinlichkeit umgerechnet:

$$\hat p_k = \frac{\exp(s_k(x))}{\sum_{j=1}^{K}\exp(s_j(x))}$$

Vorhergesagt wird die Kategorie mit der höchsten Wahrscheinlichkeit (`argmax`). Trainiert wird über die **Kreuzentropie** als Kostenfunktion, die verallgemeinerte Form des Log Loss für mehr als zwei Kategorien.

```python
# iris, LogisticRegression, train_test_split aus §16 übernommen
X_sm = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_sm = iris["target"]

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(
    X_sm, y_sm, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train_sm, y_train_sm)
print("Softmax – Score:", round(softmax_reg.score(X_test_sm, y_test_sm), 3))
print("Softmax – Vorhersage für [5, 2]:", softmax_reg.predict([[5, 2]]))
print("Softmax – Wahrscheinlichkeiten:", softmax_reg.predict_proba([[5, 2]]).round(2))
```

Scikit-Learns `LogisticRegression` verwendet intern automatisch Softmax-Regression, sobald mehr als zwei Kategorien im Trainingsdatensatz vorkommen (mit dem Standard-Solver `lbfgs`) – ein separater "SoftmaxRegression"-Klassifikator existiert nicht.

>[!important]
> Softmax-Regression sagt pro Datenpunkt immer nur **eine** Kategorie vorher (die Kategorien schließen sich gegenseitig aus). Für Aufgaben, bei denen mehrere Kategorien gleichzeitig zutreffen können (Multilabel, siehe [[Einheit 4 – Klassifikation]] §13), ist sie ungeeignet.

---

### 19. Praktische Übung zur Einheit

| Aufgabe | Inhalt |
| --- | --- |
| 1. Skalierung beim Gradientenverfahren | Trainiere ein Batch-Gradientenverfahren einmal mit skalierten und einmal mit unskalierten Merkmalen (stark unterschiedliche Wertebereiche) und vergleiche die Anzahl Epochen bis zur Konvergenz. |
| 2. Lernkurven-Diagnose bei polynomieller Regression | Erzeuge Lernkurven für polynomielle Modelle mit `degree` 1, 3 und 20 auf demselben Datensatz und ordne jede Kurve Über- oder Unteranpassung zu. |
| 3. Ridge-Regularisierungsparameter-Tuning | Trainiere `Ridge` mit mehreren `alpha`-Werten (z. B. 0, 0.01, 1, 100) auf polynomiell erweiterten Merkmalen und beschreibe, wie sich die Vorhersagekurve mit steigendem `alpha` verändert. |
| 4. Batch-GD mit Early Stopping für Softmax ohne scikit-learn | Implementiere das Batch-Gradientenverfahren für die Softmax-Regression nur mit NumPy (Kreuzentropie-Gradient aus §18) inklusive Early Stopping, und wende es auf den Iris-Datensatz an. |

---

### 20. Häufige Fehler

| Problem | Besser |
| --- | --- |
| Lernrate zu hoch gewählt → Kostenfunktion divergiert statt zu sinken | Lernrate schrittweise verkleinern (Learning Schedule) oder testweise kleinere feste Werte probieren |
| Merkmale nicht skaliert vor dem Gradientenverfahren | vor dem Training immer `StandardScaler` (o. ä.) anwenden – gilt für GD-basierte Verfahren und für Ridge/Lasso/Elastic Net gleichermaßen |
| Early Stopping ohne Kopie des besten Modells (`deepcopy`) | in jeder Epoche bei Verbesserung eine `deepcopy` des Modells sichern – sonst merkt man sich nur das *letzte*, nicht das *beste* Modell |
| Regularisierung vor Skalierung angewendet | erst skalieren, dann regularisieren – eine unskalierte Regularisierung bestraft Merkmale mit großem Wertebereich systematisch stärker |
| Polynomgrad blind erhöht, um den Trainingsfehler zu senken | Lernkurven prüfen, statt nur den Trainingsfehler zu beobachten – ein sinkender Trainingsfehler bei wachsender Lücke zum Validierungsfehler ist Overfitting, kein Fortschritt |

---

### 21. Zentrale Begriffe

| Begriff | Kurzdefinition |
| --- | --- |
| **Kostenfunktion** | Maß dafür, wie schlecht ein Modell die Trainingsdaten aktuell abbildet; wird beim Training minimiert |
| **Normalengleichung** | geschlossene Gleichung, die die optimalen Parameter einer linearen Regression direkt berechnet |
| **Gradientenverfahren** | iteratives Optimierungsverfahren, das Parameter schrittweise in Richtung sinkender Kostenfunktion anpasst |
| **Lernrate** | Hyperparameter (`eta`), der die Schrittgröße beim Gradientenverfahren steuert |
| **Batch-Gradientenverfahren** | berechnet den Gradienten bei jedem Schritt über den gesamten Trainingsdatensatz |
| **Stochastisches Gradientenverfahren** | berechnet den Gradienten bei jedem Schritt über nur einen zufälligen Datenpunkt |
| **Mini-Batch-Gradientenverfahren** | berechnet den Gradienten über kleine, zufällige Teilmengen der Trainingsdaten |
| **Lernkurve** | Diagramm von Trainings- und Validierungsfehler über der Größe des Trainingsdatensatzes; dient der Over-/Underfitting-Diagnose |
| **Ridge-Regression** | lineare Regression mit $\ell_2$-Regularisierungsterm ($\alpha$ steuert die Stärke) |
| **Lasso-Regression** | lineare Regression mit $\ell_1$-Regularisierungsterm; setzt Gewichte unwichtiger Merkmale auf null (Feature-Selection-Nebeneffekt) |
| **Elastic Net** | Mischform aus Ridge und Lasso, gesteuert über den Mischparameter `l1_ratio` |
| **Early Stopping** | Regularisierung durch Abbruch des Trainings, sobald der Validierungsfehler sein Minimum erreicht |
| **Logistische Regression** | Klassifikationsmodell, das über die sigmoide Funktion Wahrscheinlichkeiten für eine binäre Kategorie schätzt |
| **Softmax-Regression** | Verallgemeinerung der logistischen Regression auf mehr als zwei sich gegenseitig ausschließende Kategorien |
| **Entscheidungsgrenze** | Punkt bzw. Fläche im Merkmalsraum, an dem ein Klassifikator zwischen Kategorien umschaltet (Standard: $\hat p = 0{,}5$) |

---

### 22. Merksätze

>[!quote]
> Training heißt: eine Kostenfunktion minimieren – in einem Schritt (Normalengleichung) oder in vielen kleinen (Gradientenverfahren).

>[!quote]
> Skalierung ist keine Nebensache: Ohne sie wird aus der Kostenfunktions-Schüssel ein Tal, durch das das Gradientenverfahren erst einmal wandern muss.

>[!quote]
> Ein sinkender Trainingsfehler ist kein Fortschritt, solange die Lücke zum Validierungsfehler wächst.

>[!quote]
> Regularisierung ist fast nie die falsche Wahl – reine lineare Regression ohne jede Nebenbedingung ist der Sonderfall, nicht der Normalfall.

>[!quote]
> Early Stopping merkt sich das beste Modell nur, wenn man es explizit kopiert – sonst gewinnt am Ende zufällig das letzte.

---

### 23. Prüfungs- und Verständnisfragen

1. Welches Trainingsverfahren für die lineare Regression eignet sich, wenn der Trainingsdatensatz Millionen Merkmale hat, und warum?
2. Welche Algorithmen leiden unter unterschiedlich skalierten Merkmalen, und wie lässt sich das beheben?
3. Kann das Gradientenverfahren bei der logistischen Regression in einem lokalen Minimum steckenbleiben? Begründe anhand der Form der Kostenfunktion.
4. Führen Batch-, Stochastisches und Mini-Batch-Gradientenverfahren zum gleichen Modell, wenn man sie lange genug laufen lässt?
5. Was bedeutet ein während des Batch-Gradientenverfahrens stetig steigender Validierungsfehler, und wie lässt sich das Problem beheben?
6. Warum ist es keine gute Idee, das Mini-Batch-Gradientenverfahren sofort beim ersten Anstieg des Validierungsfehlers abzubrechen?
7. Welches Gradientenverfahren erreicht die Umgebung der optimalen Lösung am schnellsten, welches konvergiert am Ende exakt – und wie bringt man auch die anderen zur Konvergenz?
8. Bei einer polynomiellen Regression klafft eine große Lücke zwischen Trainings- und Validierungsfehler. Was liegt vor, und welche drei Gegenmaßnahmen gibt es?
9. Trainings- und Validierungsfehler einer Ridge-Regression sind fast identisch, aber beide recht hoch. Liegt hoher Bias oder hohe Varianz vor – und sollte `alpha` erhöht oder gesenkt werden?
10. Welche Gründe sprechen für Ridge statt einfacher linearer Regression, für Lasso statt Ridge, und für Elastic Net statt Lasso?
11. Für eine Bildklassifikation nach "innen/außen" und "Tag/Nacht" gleichzeitig: zwei logistische Regressionen oder eine Softmax-Regression – und warum?

---

### 24. Mini-Zusammenfassung

Ein lineares Regressionsmodell lässt sich entweder über die Normalengleichung in einem Rechenschritt oder iterativ über ein Gradientenverfahren trainieren – Batch-, Stochastisches und Mini-Batch-Gradientenverfahren unterscheiden sich dabei nur darin, wie viele Datenpunkte pro Schritt einbezogen werden, und damit in Geschwindigkeit, Regelmäßigkeit und Skalierbarkeit. Polynomielle Regression erweitert lineare Modelle auf nichtlineare Daten, erkauft das aber mit erhöhter Overfitting-Gefahr, die sich über Lernkurven diagnostizieren lässt – eine direkte Fortsetzung der Over-/Underfitting-Diskussion aus [[Einheit 3 – Die Machine-Learning-Umgebung]] §13. Ridge-, Lasso- und Elastic-Net-Regularisierung sowie Early Stopping liefern die passenden Gegenmaßnahmen gegen Overfitting. Dieselben Trainingsprinzipien – Kostenfunktion minimieren, Gradientenverfahren, Regularisierung – tragen schließlich auch die logistische und die Softmax-Regression für Klassifikationsaufgaben.

---

### Aufgabe

>[!important]
> Trainiere für dein eigenes QUA³CK-Projekt (oder ein geplantes Projekt) mindestens ein regularisiertes lineares Modell (Ridge, Lasso oder Elastic Net) und diagnostiziere es mit Lernkurven:
>- Trainiere zunächst ein unreguliertes lineares (oder polynomielles) Modell und plotte seine Lernkurve.
>- Trainiere anschließend ein regularisiertes Modell mit mindestens zwei verschiedenen `alpha`-Werten und vergleiche die Lernkurven.
>- Ordne das Ergebnis in Bias/Varianz-Begriffen ein: Welcher `alpha`-Wert liefert das bessere Gleichgewicht, und woran erkennst du das an den Kurven?
