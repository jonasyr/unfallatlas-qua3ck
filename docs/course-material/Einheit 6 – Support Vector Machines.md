---
title: Einheit 6 – Support Vector Machines
description: Einheit 6
date: 12-05-2026
time: 12:55
reference: Data Analytics und Big Data
index: ""
subindex: ""
status:
  - begin
---

# Einheit 6 – Support Vector Machines

>- **Reference Link:** [[Data Analytics und Big Data]]

---
>[!summary]
> Nach der linearen und logistischen Regression aus [[Einheit 5 – Trainieren von Modellen]] lernt diese Einheit mit der **Support Vector Machine (SVM)** ein zweites, grundlegend anderes Trainingsprinzip kennen: Statt eine Kostenfunktion wie den MSE oder den Log Loss zu minimieren, sucht eine SVM die **breitestmögliche Straße** zwischen den Kategorien (Large-Margin-Klassifikation).
>
> Der rote Faden: Für linear separierbare Daten genügt eine Gerade mit maximalem Abstand zu den nächsten Punkten (den *Stützvektoren*). Für nicht separierbare Daten erlaubt die **Soft-Margin-Klassifikation** kontrollierte Verletzungen dieses Randes, gesteuert über den Hyperparameter `C`. Für nichtlineare Daten hilft der **Kerneltrick**: Er berechnet das Ergebnis einer (unter Umständen unendlich-dimensionalen) Merkmalstransformation, ohne diese Transformation je tatsächlich auszuführen. Am Ende steht die gleiche SVM-Maschinerie – nur umgekehrt – auch für Regressionsaufgaben zur Verfügung.

---

### 1. Von der Kostenfunktion zur breitesten Straße

[[Einheit 5 – Trainieren von Modellen]] hat gezeigt, dass Training bedeutet, eine Kostenfunktion (MSE, Log Loss) zu minimieren. Eine **Support Vector Machine (SVM)** verfolgt für die Klassifikation ein geometrisch anderes Ziel: Sie sucht nicht die Trennlinie mit dem kleinsten Fehler, sondern diejenige, die den größtmöglichen Abstand (Margin) zu den nächstgelegenen Trainingspunkten beider Kategorien hält – man kann sich das als die **breitestmögliche Straße** zwischen den Kategorien vorstellen. Dies bezeichnet man als **Large-Margin-Klassifikation**.

>[!note]
> SVMs sind ein mächtiges und flexibles Modell: Sie bewältigen lineare wie nichtlineare Klassifikation, Regression und sogar die Erkennung von Novelties (siehe dazu vertiefend ein späteres Kapitel). Sie glänzen vor allem bei kleinen bis mittelgroßen nichtlinearen Datensätzen (etwa Hunderte bis Tausende Instanzen), skalieren aber nicht gut auf sehr große Datensätze.

Zwei Datenpunkt-Typen spielen dabei eine besondere Rolle:

- **Stützvektoren (Support Vectors)**: die Datenpunkte am Rand der Straße, die die Entscheidungsgrenze allein bestimmen ("stützen").
- Alle übrigen Punkte: Sie liegen abseits der Straße und haben **keinerlei Einfluss** auf die Entscheidungsgrenze – neue Trainingsdaten weit weg von der Straße verändern das Modell nicht.

---

### 2. Einordnung in das QUA³CK-Modell

| Phase  | Bedeutung                                                         | Rolle in dieser Einheit                                                                 |
| ------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Q**  | Question                                                          | nicht direkt betroffen                                                                   |
| **U**  | Understanding the Data                                            | Merkmalsskalierung ist für SVMs zwingend (§3) – direkte Fortsetzung von [[Einheit 5 – Trainieren von Modellen]] §6 |
| **A³** | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | SVM als weiterer Kandidat bei **Algorithm Selection**; **Adapting Features** über polynomielle bzw. ähnlichkeitsbasierte Merkmale (§6, §8); **Adjusting Hyperparameters** über `C`, `kernel`, `degree`, `gamma`, `coef0`, `epsilon` |
| **C**  | Conclude & Compare                                                | Vergleich `LinearSVC` / `SVC` / `SGDClassifier` (§10) liefert ein weiteres Entscheidungskriterium neben den Metriken aus [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]] §9 |
| **K**  | Knowledge Transfer                                                | nicht direkt betroffen                                                                   |

>[!note]
> Diese Einheit reiht sich unmittelbar hinter [[Einheit 5 – Trainieren von Modellen]] ein: Dort wurde Regularisierung über `alpha` (Ridge/Lasso/Elastic Net) eingeführt, hier erscheint mit `C` ein strukturell **umgekehrt** wirkender Regularisierungshyperparameter (kleines `C` = starke Regularisierung, großes `C` = schwache Regularisierung). Die Overfitting/Underfitting-Diagnostik aus [[Einheit 3 – Die Machine-Learning-Umgebung]] §13 bleibt unverändert gültig, nur die Stellschrauben (`C`, `gamma`, `degree`) sind neu.

---

### 3. Lernziele der Einheit

Nach dieser Einheit solltest du:

- das Prinzip der Large-Margin-Klassifikation und die Rolle der Stützvektoren erklären können
- Hard-Margin- und Soft-Margin-Klassifikation unterscheiden und den Hyperparameter `C` begründet einsetzen können
- nichtlineare SVM-Klassifikation über polynomielle Merkmale, den polynomiellen Kernel und den gaußschen RBF-Kernel anwenden können
- `LinearSVC`, `SVC` und `SGDClassifier` anhand von Geschwindigkeit, Kernel-Unterstützung und Datensatzgröße begründet auswählen können
- die SVM-Regression (SVR) und den Hyperparameter `epsilon` einordnen können
- die Zielfunktion eines linearen SVM-Klassifikators (Hard-Margin und Soft-Margin) sowie den Hinge Loss erklären können
- das duale Problem und den Kerneltrick auf konzeptioneller Ebene nachvollziehen können

>[!note]
> Zu dieser Einheit gehören zusätzlich ein Erklärvideo (*Die_breiteste_Straße__SVMs.mp4*), ein Podcast (*Support_Vector_Machines_und_die_breiteste_Straße.m4a*), ein Poster sowie ein 15-seitiger Foliensatz (*Support_Vector_Machines_Das_Bauen_der_optimalen_Straße.pdf*) und ein 62-seitiges Zusatz-Slide-Deck (*Support-Vector-Machines.pdf*). Diese Formate liegen nur als Audio/Video/Bild vor und wurden hier nicht automatisiert transkribiert – bei Bedarf manuell sichten. Der Foliensatz visualisiert exakt dieselben Inhalte wie diese Notiz, ohne inhaltlich darüber hinauszugehen. Zusätzlich liegt eine 30-Fragen-Musterlösung (*Musterloesung Large-margin Geometry.pdf*) vor, deren Kernaussagen in §22 eingearbeitet sind.

---

### 4. Lineare Klassifikation mit SVMs

Am Beispiel des Iris-Datensatzes (linear separierbare Kategorien) lässt sich das Prinzip am besten zeigen: Von mehreren Geraden, die die Kategorien korrekt trennen, wählt die SVM nicht irgendeine, sondern die mit dem größten Abstand zu den nächstgelegenen Punkten beider Kategorien – dargestellt als parallele Linien um die Entscheidungsgrenze herum.

>[!important]
> Neue Trainingsdaten *abseits* der Straße beeinflussen die Entscheidungsgrenze überhaupt nicht. Sie wird ausschließlich durch die Punkte am Rand der Straße bestimmt (oder "gestützt") – die **Stützvektoren**.

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)  # Iris virginica

svm_clf = make_pipeline(StandardScaler(),
                         LinearSVC(C=1, random_state=42))
svm_clf.fit(X, y)
```

Vorhersagen funktionieren wie gewohnt über `.predict()`. Die zugrunde liegenden vorzeichenbehafteten Abstände zur Entscheidungsgrenze liefert `.decision_function()`:

```python
X_new = [[5.5, 1.7], [5.0, 1.5]]
svm_clf.predict(X_new)
# array([ True, False])
svm_clf.decision_function(X_new)
# array([ 0.66163411, -0.22036063])
```

>[!note]
> Anders als `LogisticRegression` besitzt `LinearSVC` keine Methode `predict_proba()`. Verwendet man stattdessen die Klasse `SVC` und setzt deren Hyperparameter `probability=True`, passt scikit-learn zusätzlich ein `LogisticRegression`-Modell auf die SVM-Scores an (intern über 5-fache Kreuzvalidierung), um Wahrscheinlichkeiten zu erhalten – das verlangsamt das Training merklich.

---

### 5. Soft-Margin-Klassifikation

Verlangt man, dass **alle** Datenpunkte abseits der Straße und auf der richtigen Seite liegen, spricht man von **Hard-Margin-Klassifikation**. Sie hat zwei Probleme: Sie funktioniert nur bei linear separierbaren Daten, und sie reagiert sehr empfindlich auf Ausreißer – ein einzelner ungünstig platzierter Punkt kann das Finden eines Margins unmöglich machen oder die Trennlinie stark verschieben, sodass das Modell schlechter verallgemeinert.

**Soft-Margin-Klassifikation** löst dies durch eine Balance zwischen möglichst breiter Straße und einer begrenzten Anzahl von **Margin-Verletzungen** (Datenpunkte mitten auf der Straße oder sogar auf der falschen Seite). Gesteuert wird diese Balance über den Regularisierungshyperparameter `C`:

```python
from sklearn.svm import SVC

svm_clf = make_pipeline(StandardScaler(),
                         SVC(kernel="linear", C=1))
svm_clf.fit(X, y)
```

| `C`-Wert | Straßenbreite | Margin-Verletzungen | Risiko |
| --- | --- | --- | --- |
| niedrig (z. B. `C=1`) | breit | mehr toleriert | Underfitting bei zu niedrigem `C` |
| hoch (z. B. `C=100`) | schmal | weniger toleriert | Overfitting |

>[!important]
> `C` wirkt bei SVMs **umgekehrt** zu `alpha` bei Ridge/Lasso/Elastic Net aus [[Einheit 5 – Trainieren von Modellen]] §12: Ein **hoher** `C`-Wert bedeutet **schwache** Regularisierung (schmale Straße, wenige Verletzungen), ein **niedriger** `C`-Wert **starke** Regularisierung (breite Straße, mehr Verletzungen). Overfittet Ihr SVM-Modell, senken Sie `C`.

>[!warning]
> SVMs reagieren empfindlich auf die Skalierung der Merkmale: Ist ein Merkmal in seinem Wertebereich deutlich größer als ein anderes, wird die breitestmögliche Straße fast ausschließlich von diesem einen Merkmal dominiert. Nach dem Skalieren (z. B. mit `StandardScaler`) sieht die Entscheidungsgrenze deutlich sinnvoller aus – SVMs also grundsätzlich nur auf skalierten Merkmalen trainieren.

---

### 6. Nichtlineare SVM-Klassifikation: Polynomielle Merkmale

Auch wenn lineare SVM-Klassifikatoren sehr effizient sind, sind viele Datensätze nicht einmal annähernd linear separierbar. Eine Möglichkeit im Umgang mit nichtlinearen Datensätzen ist das Hinzufügen zusätzlicher Merkmale wie polynomieller Merkmale (wie in [[Einheit 5 – Trainieren von Modellen]] §10) – dabei kann ein linear nicht separierbarer Datensatz mit einem Merkmal $x_1$ durch Hinzufügen von $x_2 = (x_1)^2$ linear separierbar werden.

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)
)
polynomial_svm_clf.fit(X, y)
```

Der Datensatz *moons* (zwei ineinander verschränkte Halbkreise) lässt sich mit dieser Pipeline (`PolynomialFeatures` → `StandardScaler` → `LinearSVC`) linear im erweiterten Merkmalsraum, aber nichtlinear im ursprünglichen Merkmalsraum trennen.

---

### 7. Der polynomielle Kernel

Das Hinzufügen polynomieller Merkmale ist einfach umzusetzen und funktioniert bei vielen Machine-Learning-Algorithmen gut – niedrige Polynomgrade kommen jedoch mit komplexen Datensätzen nicht gut zurecht, höhere Grade erzeugen dagegen eine riesige Merkmalsanzahl und machen das Modell zu langsam.

Bei SVMs löst der **Kerneltrick** dieses Dilemma: Er ermöglicht es, das gleiche Ergebnis wie beim expliziten Hinzufügen vieler polynomieller Merkmale zu erhalten, ohne diese Merkmale tatsächlich zu berechnen – auch bei sehr hohen Polynomgraden.

```python
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="poly", degree=3, coef0=1, C=5)
)
poly_kernel_svm_clf.fit(X, y)
```

| Hyperparameter | Bedeutung |
| --- | --- |
| `degree` | Grad der Polynome; bei Overfitting senken, bei Underfitting erhöhen |
| `coef0` | steuert, wie stark das Modell von höhergradigen im Gegensatz zu niedriggradigen Termen beeinflusst wird |

---

### 8. Ähnlichkeitsbasierte Merkmale und Landmarken

Eine weitere Technik im Umgang mit nichtlinearen Daten besteht darin, mit einer **Ähnlichkeitsfunktion** berechnete Merkmale hinzuzufügen. Diese Funktion misst, wie ähnlich ein Datenpunkt zu einer festgelegten **Landmarke** ist.

Als Ähnlichkeitsfunktion dient häufig die gaußsche **radiale Basisfunktion (RBF)**:

$$\phi(\mathbf{x}, \ell) = \exp(-\gamma \|\mathbf{x} - \ell\|^2)$$

eine glockenförmige Funktion zwischen 0 (sehr weit von der Landmarke entfernt) und 1 (genau an der Landmarke). Fügt man z. B. bei einem eindimensionalen Datensatz zwei Landmarken hinzu, wird ein zuvor nicht separierbarer Datensatz im transformierten (zweidimensionalen) Merkmalsraum linear separierbar.

>[!note]
> Die einfachste Art, Landmarken auszuwählen, ist, bei jedem einzelnen Datenpunkt eine Landmarke zu erzeugen. Ein Datensatz mit $m$ Punkten und $n$ Merkmalen wird so in einen Trainingsdatensatz mit $m$ Punkten und $m$ Merkmalen umgewandelt (bei Verwerfen der ursprünglichen Merkmale) – bei großen Datensätzen entsteht so eine entsprechend große Merkmalsanzahl.

---

### 9. Der gaußsche RBF-Kernel

Wie bei den polynomiellen Merkmalen kann auch das explizite Berechnen aller ähnlichkeitsbasierten Merkmale sehr rechenintensiv werden. Der Kerneltrick ermöglicht auch hier, ein äquivalentes Ergebnis zu erhalten, ohne die zusätzlichen Merkmale tatsächlich zu berechnen:

```python
rbf_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", gamma=5, C=0.001)
)
rbf_kernel_svm_clf.fit(X, y)
```

`gamma` ($\gamma$) verhält sich wie ein zweiter Regularisierungsparameter neben `C`:

| `gamma` | Effekt auf die Glockenkurve | Effekt auf die Entscheidungsgrenze | bei ... erhöhen/senken |
| --- | --- | --- | --- |
| groß | schmal | unregelmäßig, schlängelt sich um einzelne Punkte | Overfitting → senken |
| klein | breit | weicher, glatter | Underfitting → erhöhen |

>[!tip]
> Als Faustregel gilt: Immer zuerst den linearen Kernel ausprobieren (`LinearSVC` ist deutlich schneller als `SVC(kernel="linear")`, besonders bei umfangreichen Trainingsdaten oder sehr vielen Merkmalen). Reicht das nicht aus, mit dem gaußschen RBF-Kernel weitermachen – er funktioniert in vielen Fällen ziemlich gut. Es gibt zudem weitere, seltener genutzte Kernels (z. B. String-Kernels für Textdokumente oder DNA-Sequenzen), die bei entsprechend spezialisierten Datenstrukturen eine Chance verdienen.

---

### 10. SVM-Klassen und Rechenkomplexität

| Klasse | Zeitliche Komplexität | Out-of-Core möglich | Scaling nötig | Kerneltrick |
| --- | --- | --- | --- | --- |
| `LinearSVC` | $O(m \times n)$ | nein | ja | nein |
| `SVC` | $O(m^2 \times n)$ bis $O(m^3 \times n)$ | nein | ja | ja |
| `SGDClassifier` | $O(m \times n)$ | ja | ja | nein |

- **`LinearSVC`** verwendet die Bibliothek *liblinear*, unterstützt den Kerneltrick nicht, skaliert aber annähernd linear mit der Größe des Trainingsdatensatzes und der Anzahl Merkmale.
- **`SVC`** verwendet die Bibliothek *libsvm* und unterstützt den Kerneltrick. Sie wird bei großen Trainingsdatensätzen (Hunderttausende Datenpunkte) unakzeptabel langsam, eignet sich aber ausgezeichnet für kleine und mittelgroße nichtlineare Datensätze, insbesondere bei dünn besetzten Merkmalen.
- **`SGDClassifier`** führt standardmäßig ebenfalls eine große Margin-Klassifikation durch und lässt sich über `alpha`, `penalty` und `learning_rate` ähnlich wie eine lineare SVM konfigurieren (Training über stochastisches Gradientenverfahren, siehe [[Einheit 5 – Trainieren von Modellen]] §8). Sie ermöglicht inkrementelles Lernen und Out-of-Core-Learning bei geringem Speichereinsatz.

>[!tip]
> Praktische Faustregel: Mit `LinearSVC` starten (schnelle Baseline). Reicht Linearität nicht aus, `SVC` mit `kernel="rbf"` verwenden. Passt der Datensatz nicht in den Speicher oder wird inkrementelles Lernen benötigt, ist `SGDClassifier` mit Hinge Loss die passende Wahl.

---

### 11. SVM-Regression

Statt SVMs zur Klassifikation einzusetzen, lässt sich das Prinzip auch für Regression umkehren: Statt die breitestmögliche Straße zwischen zwei Kategorien zu fitten, versucht die **SVM-Regression (SVR)**, so viele Datenpunkte wie möglich *auf* die Straße zu bringen und Grenzverletzungen (hier: Punkte *abseits* der Straße) zu minimieren. Die Breite der Straße wird über den Hyperparameter `epsilon` ($\varepsilon$) gesteuert.

```python
from sklearn.svm import LinearSVR

svm_reg = make_pipeline(StandardScaler(),
                         LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(X, y)
```

Ein Verringern von `epsilon` vergrößert die Anzahl an Stützvektoren, was zu einer Regularisierung des Modells führt. Trainingsdatenpunkte *innerhalb* des Margins beeinflussen die Vorhersagen des Modells nicht – man bezeichnet dieses Modell daher als $\varepsilon$-insensitiv.

Für nichtlineare Regressionsaufgaben kommt analog zur Klassifikation ein Kernel-SVM-Modell zum Einsatz:

```python
from sklearn.svm import SVR

svm_poly_reg = make_pipeline(StandardScaler(),
                              SVR(kernel="poly", degree=2, C=100, epsilon=0.1))
svm_poly_reg.fit(X, y)
```

| Klasse | Pendant zu | Skalierungsverhalten |
| --- | --- | --- |
| `LinearSVR` | `LinearSVC` | skaliert linear mit der Größe des Trainingsdatensatzes |
| `SVR` | `SVC` | wird mit stark wachsendem Trainingsdatensatz sehr langsam |

---

### 12. Hinter den Kulissen: Entscheidungsfunktion und Hard-Margin-Zielfunktion

Ein linearer SVM-Klassifikator sagt die Kategorie eines neuen Datenpunkts $\mathbf{x}$ vorher, indem er zuerst die Entscheidungsfunktion $\mathbf{w}^\top \mathbf{x} + b$ berechnet: Ist das Ergebnis positiv, wird die positive Kategorie ($\hat y = 1$) vorhergesagt, andernfalls die negative ($\hat y = 0$) – wie bei `LogisticRegression` (siehe [[Einheit 4 – Klassifikation]]).

>[!note]
> Konvention in diesem Kapitel: Der Bias-Term $b$ und der Gewichtsvektor $\mathbf{w}$ (mit den Gewichten $w_1$ bis $w_n$, ohne Bias-Merkmal) werden getrennt behandelt, statt wie in [[Einheit 5 – Trainieren von Modellen]] alle Parameter in einen Vektor $\theta$ zu packen.

Um die Straße möglichst breit zu machen, muss $\mathbf{w}$ möglichst klein gehalten werden – der Bias-Term $b$ hat dagegen **keinen** Einfluss auf die Breite des Margins, er verschiebt den Margin nur, ohne seine Größe zu verändern. Damit keine Margin-Verletzungen auftreten, muss die Entscheidungsfunktion für alle positiven Trainingspunkte $\geq 1$ und für alle negativen $\leq -1$ sein. Mit $t^{(i)} = -1$ für negative und $t^{(i)} = 1$ für positive Punkte lässt sich dies einheitlich als $t^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1$ ausdrücken.

Damit ergibt sich die Zielfunktion eines linearen SVM-Klassifikators mit **Hard-Margin**:

$$\underset{\mathbf{w},b}{\text{minimiere}} \quad \frac{1}{2}\mathbf{w}^\top \mathbf{w} \qquad \text{unter der Bedingung} \quad t^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1 \ \text{für}\ i = 1, \dots, m$$

>[!note]
> Minimiert wird $\tfrac{1}{2}\mathbf{w}^\top \mathbf{w}$ statt $\|\mathbf{w}\|$: Beide werden am selben Punkt minimal, aber $\tfrac{1}{2}\|\mathbf{w}\|^2$ besitzt eine einfache, überall stetige Ableitung, während $\|\mathbf{w}\|$ bei $\mathbf{w}=0$ nicht differenzierbar ist – Optimierungsalgorithmen laufen mit differenzierbaren Funktionen zuverlässiger.

---

### 13. Soft-Margin-Zielfunktion, Slack-Variablen und Hinge Loss

Um die Zielfunktion für Soft-Margin zu erhalten, wird für jeden Datenpunkt $i$ eine nichtnegative **Slack-Variable** $\zeta^{(i)} \geq 0$ eingeführt: Sie bestimmt, wie stark Punkt $i$ den Margin verletzen darf. Es entstehen zwei gegenläufige Ziele – die Slack-Variablen möglichst klein halten (wenig Verletzungen) und $\tfrac{1}{2}\mathbf{w}^\top\mathbf{w}$ möglichst klein halten (breiter Margin). Der Hyperparameter `C` legt die Balance zwischen beiden Zielen fest:

$$\underset{\mathbf{w},b,\zeta}{\text{minimiere}} \quad \frac{1}{2}\mathbf{w}^\top \mathbf{w} + C\sum_{i=1}^{m}\zeta^{(i)} \qquad \text{unter}\quad t^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1-\zeta^{(i)} \ \text{und}\ \zeta^{(i)} \geq 0$$

Sowohl das Hard-Margin- als auch das Soft-Margin-Problem sind konvexe **quadratische Optimierungsprobleme mit linearen Nebenbedingungen** (kurz: *quadratische Programme*, QP), für die es zahlreiche fertige Solver gibt.

Eine alternative Trainingsmethode ist der Einsatz des Gradientenverfahrens zum Minimieren des **Hinge Loss** (oder des **Squared Hinge Loss**). Für einen Datenpunkt $\mathbf{x}$ der positiven Kategorie ($t=1$) ist der Hinge Loss null, sobald der Score $s = \mathbf{w}^\top\mathbf{x}+b \geq 1$ ist (Punkt außerhalb der Straße auf der richtigen Seite); analog für $t=-1$ und $s \leq -1$:

$$\text{Hinge Loss} = \max(0,\, 1-t\cdot s)$$

| Loss | Wachstum bei Verletzung | Ausreißerempfindlichkeit | Standard bei |
| --- | --- | --- | --- |
| Hinge Loss | linear | geringer | `SGDClassifier` |
| Squared Hinge Loss | quadratisch | größer | `LinearSVC` |

Je weiter weg ein Datenpunkt von der korrekten Seite des Margins ist, desto größer der Verlust: Beim Hinge Loss wächst er linear, beim Squared Hinge Loss quadratisch – Letzterer reagiert damit empfindlicher auf Ausreißer, tendiert aber dazu, schneller zu konvergieren. Beide Klassen lassen sich über den Hyperparameter `loss` (`"hinge"` oder `"squared_hinge"`) umstellen; der Optimierungsalgorithmus von `SVC` findet eine ähnliche Lösung wie beim Minimieren des Hinge Loss.

---

### 14. Das duale Problem

Das bisher betrachtete Optimierungsproblem heißt **primales Problem**. Zu jedem primalen Optimierungsproblem lässt sich ein eng verwandtes **duales Problem** formulieren. Bei SVMs haben primales und duales Problem die gleiche Lösung, sodass man sich aussuchen kann, welches man löst:

$$\underset{\alpha}{\text{minimiere}} \quad \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha^{(i)}\alpha^{(j)}t^{(i)}t^{(j)}\mathbf{x}^{(i)\top}\mathbf{x}^{(j)} - \sum_{i=1}^{m}\alpha^{(i)} \quad \text{unter } \alpha^{(i)} \geq 0 \ \text{und} \ \sum_{i=1}^m \alpha^{(i)}t^{(i)}=0$$

Hat man den Vektor $\hat{\boldsymbol\alpha}$ gefunden, der diese Gleichung minimiert (mit einem QP-Solver), lassen sich $\hat{\mathbf{w}}$ und $\hat b$ daraus zurückrechnen, wobei $n_s$ für die Anzahl der Stützvektoren steht:

$$\hat{\mathbf{w}} = \sum_{i=1}^{m}\alpha^{(i)}t^{(i)}\mathbf{x}^{(i)}, \qquad \hat b = \frac{1}{n_s}\sum_{\substack{i=1\\ \hat\alpha^{(i)}>0}}^{m}\left(t^{(i)} - \hat{\mathbf{w}}^\top \mathbf{x}^{(i)}\right)$$

>[!note]
> Das duale Problem lässt sich schneller als das primale lösen, wenn die Anzahl Trainingspunkte kleiner als die Anzahl der Merkmale ist. Bedeutender ist aber: Das duale Problem ermöglicht den Kerneltrick, was mit dem primalen Problem nicht funktioniert.

---

### 15. Kernel-SVM: Wie der Kerneltrick funktioniert

Möchte man auf zweidimensionalen Trainingsdaten eine polynomielle Transformation 2. Grades $\phi$ durchführen und anschließend einen linearen SVM-Klassifikator auf den transformierten Daten trainieren:

$$\phi(\mathbf{x}) = \phi\left(\begin{pmatrix}x_1\\x_2\end{pmatrix}\right) = \begin{pmatrix}x_1^2\\\sqrt2\, x_1 x_2\\x_2^2\end{pmatrix}$$

Berechnet man für zwei zweidimensionale Vektoren $\mathbf{a}$ und $\mathbf{b}$ diese Transformation und anschließend das Skalarprodukt der transformierten Vektoren, ergibt sich eine entscheidende Erkenntnis:

$$\phi(\mathbf{a})^\top\phi(\mathbf{b}) = (\mathbf{a}^\top\mathbf{b})^2$$

Das Skalarprodukt der transformierten Vektoren entspricht exakt dem **Quadrat** des Skalarprodukts der ursprünglichen Vektoren – man braucht die Trainingsdaten also überhaupt nicht zu transformieren: Da das duale Problem (§14) nur Skalarprodukte $\phi(\mathbf{x}^{(i)})^\top\phi(\mathbf{x}^{(j)})$ enthält, genügt es, dieses Skalarprodukt einfach durch $(\mathbf{x}^{(i)\top}\mathbf{x}^{(j)})^2$ zu ersetzen. Das Ergebnis ist exakt das gleiche, als hätte man die Trainingsdaten tatsächlich transformiert – der gesamte Prozess wird dadurch rechnerisch wesentlich effizienter.

Die Funktion $K(\mathbf{a},\mathbf{b}) = (\mathbf{a}^\top\mathbf{b})^2$ nennt man einen **polynomiellen Kernel 2. Grades**. Allgemein versteht man unter einem **Kernel** eine Funktion, mit der sich das Skalarprodukt $\phi(\mathbf{a})^\top\phi(\mathbf{b})$ lediglich aus den ursprünglichen Vektoren $\mathbf{a}$ und $\mathbf{b}$ berechnen lässt, ohne dass die Transformation $\phi$ überhaupt bekannt sein muss:

| Kernel | Formel |
| --- | --- |
| Linear | $K(\mathbf{a},\mathbf{b}) = \mathbf{a}^\top\mathbf{b}$ |
| Polynomiell | $K(\mathbf{a},\mathbf{b}) = (\gamma\mathbf{a}^\top\mathbf{b}+r)^d$ |
| Gaußsche RBF | $K(\mathbf{a},\mathbf{b}) = \exp(-\gamma\|\mathbf{a}-\mathbf{b}\|^2)$ |
| Sigmoid | $K(\mathbf{a},\mathbf{b}) = \tanh(\gamma\mathbf{a}^\top\mathbf{b}+r)$ |

---

### 16. Mercers Theorem

Laut **Mercers Theorem** muss, wenn eine Funktion $K(\mathbf{a},\mathbf{b})$ bestimmte mathematische Bedingungen erfüllt (die *Mercer-Bedingungen*: $K$ muss stetig sein, seine Parameter symmetrisch, sodass $K(\mathbf{a},\mathbf{b})=K(\mathbf{b},\mathbf{a})$ gilt, und die Funktion muss eine positiv semidefinite Gram-Matrix erzeugen), auch eine Funktion $\phi$ existieren, die $\mathbf{a}$ und $\mathbf{b}$ in einen anderen (möglicherweise sehr viel höherdimensionalen) Raum abbildet, sodass gilt: $K(\mathbf{a},\mathbf{b}) = \phi(\mathbf{a})^\top\phi(\mathbf{b})$.

Im Fall des gaußschen RBF-Kernels lässt sich zeigen, dass $\phi$ jeden Trainingsdatenpunkt in einen Raum mit **unendlich vielen** Dimensionen transformiert – man muss diese Zuordnung aber, wie gesehen, nie tatsächlich vornehmen.

>[!note]
> Manche häufig eingesetzten Kernels (etwa der sigmoide Kernel) erfüllen nicht alle Mercer-Bedingungen. In der Praxis funktionieren sie dennoch oft gut genug – die formale Garantie fehlt in diesen Fällen aber.

---

### 17. Vorhersagen mit einer Kernel-SVM treffen

Um ein loses Ende zu schließen: Der duale-zu-primal-Übergang aus §14 setzt eigentlich voraus, $\hat{\mathbf{w}}$ zu kennen – bei Verwendung des Kerneltricks hätte $\hat{\mathbf{w}}$ aber die gleiche (möglicherweise unendliche) Dimension wie $\phi(\mathbf{x}^{(i)})$ und ist damit nicht berechenbar. Setzt man die Formel für $\hat{\mathbf{w}}$ jedoch direkt in die Entscheidungsfunktion für einen neuen Datenpunkt $\mathbf{x}^{(n)}$ ein, erhält man eine Formel, die ausschließlich aus Skalarprodukten zwischen Eingabevektoren besteht – der Kerneltrick lässt sich also erneut anwenden:

$$h_{\hat{\mathbf{w}},\hat b}(\phi(\mathbf{x}^{(n)})) = \sum_{\substack{i=1\\ \hat\alpha^{(i)}>0}}^{m}\hat\alpha^{(i)}t^{(i)}K(\mathbf{x}^{(i)},\mathbf{x}^{(n)}) + \hat b$$

Weil $\alpha^{(i)} \neq 0$ nur für die Stützvektoren gilt, muss für Vorhersagen das Skalarprodukt (hier: der Kernel) nur zwischen dem neuen Punkt und den Stützvektoren berechnet werden – nicht mit sämtlichen Trainingspunkten. Die Rechenzeit einer Vorhersage skaliert damit direkt mit der **Anzahl der Stützvektoren**, nicht mit der Größe des gesamten Trainingsdatensatzes.

>[!note]
> Es ist ebenfalls möglich, Online-Kernel-SVMs zu implementieren (inkrementelles Lernen). Für größere nichtlineare Aufgaben sollten Sie allerdings eher Random Forests oder neuronale Netze in Betracht ziehen.

---

### 18. Praktische Übung zur Einheit

| Aufgabe | Inhalt |
| --- | --- |
| 1. `LinearSVC` vs. `SVC` vs. `SGDClassifier` | Trainiere einen `LinearSVC` auf linear separierbaren Daten. Trainiere anschließend einen `SVC` (linearer Kernel) und einen `SGDClassifier` auf dem gleichen Datensatz und vergleiche, ob alle drei ein etwa gleiches Modell berechnen. |
| 2. SVM-Klassifikator auf dem Wein-Datensatz | Trainiere einen SVM-Klassifikator auf `sklearn.datasets.load_wine()`. Da SVM-Klassifikatoren binär sind, setze die One-versus-All-Strategie aus [[Einheit 4 – Klassifikation]] §11 ein, um alle drei Kategorien zu klassifizieren. Welche Genauigkeit erreichst du? |
| 3. Hyperparameter-Wirkung visualisieren | Trainiere auf dem *moons*-Datensatz vier `SVC(kernel="rbf")`-Modelle mit den Kombinationen `gamma∈{0.1, 5}` × `C∈{0.001, 1000}` und ordne jede Entscheidungsgrenze Über- oder Unteranpassung zu. |
| 4. SVM-Regressor auf Immobiliendaten | Trainiere und optimiere einen SVM-Regressor auf `sklearn.datasets.fetch_california_housing()`. Da SVMs bei über 20.000 Datenpunkten langsam werden, nutze für die Hyperparameteroptimierung deutlich weniger Datenpunkte (z. B. 2.000). Wie groß ist der beste RMSE deines Modells? |

---

### 19. Häufige Fehler

| Problem | Besser |
| --- | --- |
| Merkmale nicht skaliert vor dem SVM-Training | vor jedem SVM-Training immer `StandardScaler` (o. ä.) in einer Pipeline anwenden – gilt für lineare wie für Kernel-SVMs gleichermaßen |
| `C` wie `alpha` behandelt (hoher Wert = starke Regularisierung angenommen) | `C` wirkt umgekehrt zu `alpha`: hohes `C` = schwache Regularisierung. Bei Overfitting `C` senken, bei Underfitting `C` erhöhen |
| `gamma` beim RBF-Kernel blind erhöht, um Underfitting zu beheben, ohne `C` zu beachten | `C` und `gamma` gemeinsam betrachten – beide wirken gleichsinnig auf die Modellkomplexität (Overfitting → beide senken, Underfitting → beide erhöhen) |
| `SVC(kernel="rbf")` direkt auf sehr große Trainingsdatensätze (Hunderttausende Zeilen) angewendet | bei großem $m$ zuerst `LinearSVC` oder `SGDClassifier` prüfen; `SVC` skaliert zwischen $O(m^2 n)$ und $O(m^3 n)$ und wird dann unpraktikabel langsam |
| Angenommen, weit entfernte, korrekt klassifizierte Punkte verändern die SVM-Entscheidungsgrenze | nur Stützvektoren (Punkte auf oder innerhalb des Margins) besitzen $\alpha^{(i)} \neq 0$ und beeinflussen die Lösung – weit entfernte Punkte bleiben folgenlos |

---

### 20. Zentrale Begriffe

| Begriff | Kurzdefinition |
| --- | --- |
| **Support Vector Machine (SVM)** | Modell, das eine Entscheidungsgrenze mit maximalem Abstand (Margin) zu den nächstgelegenen Trainingspunkten sucht |
| **Large-Margin-Klassifikation** | Prinzip, die "breitestmögliche Straße" zwischen den Kategorien zu finden |
| **Stützvektor (Support Vector)** | Trainingspunkt am Rand des Margins, der die Entscheidungsgrenze bestimmt |
| **Hard-Margin-Klassifikation** | SVM-Variante, die verlangt, dass alle Punkte korrekt und abseits der Straße liegen |
| **Soft-Margin-Klassifikation** | SVM-Variante, die kontrollierte Margin-Verletzungen erlaubt (Hyperparameter `C`) |
| **Slack-Variable ($\zeta$)** | nichtnegative Variable, die das Ausmaß der Margin-Verletzung eines Punkts misst |
| **Hinge Loss** | Verlustfunktion $\max(0, 1-ts)$ zum gradientenbasierten Training linearer SVMs |
| **Kerneltrick** | Berechnung des Skalarprodukts transformierter Merkmale, ohne die Transformation je auszuführen |
| **Polynomieller Kernel** | Kernel der Form $(\gamma \mathbf{a}^\top\mathbf{b}+r)^d$ |
| **Gaußscher RBF-Kernel** | Kernel der Form $\exp(-\gamma\|\mathbf{a}-\mathbf{b}\|^2)$; entspricht ähnlichkeitsbasierten Merkmalen mit unendlich vielen Landmarken |
| **`gamma` ($\gamma$)** | Hyperparameter des RBF- bzw. polynomiellen Kernels; wirkt wie ein zweiter Regularisierungsparameter |
| **Primales Problem** | ursprüngliches SVM-Optimierungsproblem über $\mathbf{w}$ und $b$ |
| **Duales Problem** | äquivalentes Optimierungsproblem über die Lagrange-Multiplikatoren $\alpha$; ermöglicht den Kerneltrick |
| **Mercers Theorem** | mathematisches Kriterium, das garantiert, dass eine Kernel-Funktion einem Skalarprodukt in einem (möglicherweise unendlichdimensionalen) Merkmalsraum entspricht |
| **SVM-Regression (SVR)** | Umkehrung der SVM-Klassifikation: möglichst viele Punkte auf die Straße bringen statt sie zu trennen |
| **`epsilon` ($\varepsilon$)** | Hyperparameter der SVR; Breite der Straße, innerhalb derer Fehler nicht bestraft werden |

---

### 21. Merksätze

>[!quote]
> Eine SVM sucht nicht die Trennlinie mit dem kleinsten Fehler, sondern die mit dem größten Abstand zu den nächsten Punkten.

>[!quote]
> Nur die Stützvektoren zählen: Alles, was abseits der Straße liegt, hätte auch ganz woanders liegen können, ohne das Modell zu verändern.

>[!quote]
> `C` bei SVMs verhält sich spiegelverkehrt zu `alpha` bei Ridge & Co.: groß heißt hier schwach reguliert, nicht stark.

>[!quote]
> Der Kerneltrick berechnet das Ergebnis einer Transformation, ohne die Transformation je auszuführen – und macht so auch unendlich-dimensionale Merkmalsräume praktisch nutzbar.

>[!quote]
> Skalierung ist bei SVMs keine Kür, sondern Pflicht: Ohne sie dominiert das Merkmal mit dem größten Wertebereich die gesamte Straße.

---

### 22. Prüfungs- und Verständnisfragen

1. Was ist die den Support Vector Machines zugrunde liegende Idee, und was ist ein Stützvektor?
2. Warum ist es wichtig, beim Verwenden von SVMs die Eingabedaten zu skalieren?
3. Kann ein SVM-Klassifikator einen Konfidenzwert bzw. eine Wahrscheinlichkeit für seine Vorhersage ausgeben? Wie unterscheiden sich `LinearSVC` und `SVC(probability=True)` dabei?
4. Wie können Sie sich zwischen `LinearSVC`, `SVC` und `SGDClassifier` entscheiden?
5. Ein RBF-Kernel-SVM scheint bei den Trainingsdaten zu underfitten – sollten Sie `gamma` erhöhen oder senken? Wie sieht es mit `C` aus?
6. Was bedeutet es für ein SVR-Modell, $\varepsilon$-insensitiv zu sein?
7. Wofür wird der Kerneltrick eingesetzt, und warum funktioniert er nur im dualen, nicht im primalen Problem?
8. In einer Soft-Margin-SVM mit $m=500$ Trainingspunkten und $n=5000$ Merkmalen: Ist es günstiger, das primale oder das duale Problem zu lösen, und warum?
9. Ihre Daten sind fast linear separierbar, enthalten aber einige klare Ausreißer. Welche SVM-Konfiguration (Hard-/Soft-Margin, `C`-Wert) ist am besten geeignet, und warum?
10. Worin unterscheiden sich Hinge Loss und Squared Hinge Loss, und welche Klasse verwendet welchen standardmäßig?
11. Warum bleibt die Margin-Breite unverändert, wenn Sie bei fixiertem $\mathbf{w}$ nur den Bias-Term $b$ verändern?
12. Ein RBF-SVM overfittet: Die Trainingsgenauigkeit ist hoch, die Validierungsgenauigkeit niedrig. Welche gemeinsame Anpassung von `C` und `gamma` ist am ehesten hilfreich?

---

### 23. Mini-Zusammenfassung

Eine Support Vector Machine sucht statt einer kostenminimalen Lösung die Entscheidungsgrenze mit dem größtmöglichen Abstand zu den nächstgelegenen Trainingspunkten – bestimmt ausschließlich durch die Stützvektoren. Die Hard-Margin-Variante verlangt perfekte Trennbarkeit ohne Verletzungen, die in der Praxis übliche Soft-Margin-Variante erlaubt über den Hyperparameter `C` kontrollierte Verletzungen (`C` wirkt dabei umgekehrt zu `alpha` aus [[Einheit 5 – Trainieren von Modellen]]). Für nichtlineare Daten liefern polynomielle bzw. ähnlichkeitsbasierte Merkmale eine Lösung, die der Kerneltrick – ermöglicht durch das duale Optimierungsproblem – rechnerisch effizient macht, ohne die Merkmalstransformation je auszuführen. `LinearSVC`, `SVC` und `SGDClassifier` unterscheiden sich vor allem in Rechenkomplexität, Kernel-Unterstützung und Eignung für sehr große Datensätze. Dasselbe Grundprinzip lässt sich über die SVM-Regression (Hyperparameter `epsilon`) auch auf Regressionsaufgaben anwenden.

---

### Aufgabe

>[!important]
> Trainiere für dein eigenes QUA³CK-Projekt (oder ein geplantes Projekt) mindestens einen SVM-Klassifikator und vergleiche ihn systematisch mit einem Modell aus [[Einheit 5 – Trainieren von Modellen]]:
>- Trainiere zunächst einen `LinearSVC` (in einer Pipeline mit `StandardScaler`) als Baseline und bewerte ihn mit den Metriken aus [[Einheit 1 – QUA³CK-Prozessmodell für Data-Science-Projekte]] §9.
>- Trainiere anschließend einen `SVC(kernel="rbf")` mit mindestens zwei verschiedenen `gamma`/`C`-Kombinationen und vergleiche die Ergebnisse.
>- Ordne den Vergleich in Bias/Varianz-Begriffen ein: Welche Konfiguration liefert das bessere Gleichgewicht, und woran erkennst du das (Trainings- vs. Validierungsmetrik, Anzahl Stützvektoren)?
>- Vergleiche abschließend kurz Trainingsdauer und Ergebnisqualität mit einem logistischen Regressionsmodell aus [[Einheit 5 – Trainieren von Modellen]] §16 auf denselben Daten.
