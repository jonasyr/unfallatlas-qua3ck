# Classification

> **Summary:**
>
> This file is a condensed version of the A³ phase (Algorithm Selection): A binary classifier is trained on the MNIST dataset, evaluated using cross-validation and analyzed using a range of quality metrics that go far beyond mere accuracy. The common thread: Accuracy alone is a trap as soon as classes are unbalanced - a point that was already touched upon in [docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md](docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md) `## 8 C - Conclude & Compare: Evaluate models` and is finally fleshed out here with the confusion matrix, precision, sensitivity, F1 score, ROC curve and AUC.

**ToC:**
- [1 Why do you need more than one Accuracy score?](#1-why-do-you-need-more-than-one-accuracy-score)
- [2 Classification within the QUA³CK model](#2-classification-within-the-qua³ck-model)
- [3 MNIST: The Dataset](#3-mnist-the-dataset)
- [4 Training a binary classifier](#4-training-a-binary-classifier)
- [5 Performance measurement with cross-validation](#5-performance-measurement-with-cross-validation)
- [6 The Confusion Matrix](#6-the-confusion-matrix)
- [7 Precision and sensitivity](#7-precision-and-sensitivity)
- [8 The precision/recall trade-off](#8-the-precisionrecall-trade-off)
- [9 The ROC Curve](#9-the-roc-curve)
- [10 Multiclass classification](#10-multiclass-classification)
- [11 Error analysis](#11-error-analysis)
- [12 Multilabel classification](#12-multilabel-classification)
- [13 Multioutput classification](#13-multioutput-classification)
- [14 Key terms](#14-key-terms)
- [15 Memorable quotes](#15-memorable-quotes)

---

## 1 Why do you need more than one Accuracy score?

Classification - predicting discrete categories rather than numerical values - is considered, alongside regression, the most common task in supervised learning. Evaluating a classifier is often much trickier than evaluating a regressor: a single percentage for "correctly classified" can easily obscure which errors a model is making and how costly those errors are in practice.

**Note:** This file uses the MNIST dataset throughout - 70,000 small images of handwritten digits that are considered the "Hello World" of machine learning. Every new classification method is measured against MNIST sooner or later.

---

## 2 Classification within the QUA³CK model

| **Phase** | **Meaning**                                                       | **Role in this file**                                                                           |
| --------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Q**     | Question                                                          | Not directly                                                                                    |
| **U**     | Understanding the Data                                            | MNIST must be understood before training: image size, pixel intensities and class distribution. |
| **A³**    | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | Core focus of this unit: SGD, Random Forest, SVM and KNN are selected, compared, and evaluated. |
| **C**     | Conclude & Compare                                                | the same metrics as in 002_QUACK_prozess_model_for_Data_Science_Projects.md                     |
| **K**     | Knowledge Transfer                                                | Not directly                                                                                    |

**Note:** File [docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md](docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md) lists in `## 8` accuracy, precision, recall and the F1 score as quantitative criteria for the C phase, without explaining in detail how they are calculated or when each metric is actually useful. This file fills exactly that gap.

---

## 3 MNIST: The Dataset

MNIST consists of 70,000 images of handwritten digits (0-9), collected from U.S. high school students and Census Bureau employees. Each image has 28 × 28 pixels, or 784 features - one per pixel, with values from 0 (white) to 255 (black).

```python
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target.astype(np.uint8)  # Labels as Integer

# The dataset already comes shuffled and split:
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
```

**Important:** The test dataset is set aside before the data is examined more closely - exactly as already required in [docs/course-material/003_Understanding_the_data.md](docs/course-material/003_Understanding_the_data.md). The fact that the training data is already shuffled is no coincidence: in cross-validation, all folds should have a similar class distribution; otherwise, some folds would be missing entire digits.

---

## 4 Training a binary classifier

To simplify the task, we will first build only a binary classifier: "Is this a 5 or not?". A solid starting point is the `SGDClassifier` (stochastic gradient descent), which processes training data points one at a time and is therefore also suitable for very large datasets and online learning.

```python
from sklearn.linear_model import SGDClassifier

y_train_5 = (y_train == 5)  # True for all 5s, False for all others
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([X_train[0]])  # array([True]) – correct, X_train[0] is a 5
```

---

## 5 Performance measurement with cross-validation

**Cross-validation also provides the usual accuracy values here, which look impressive at first glance:**
```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.95035, 0.96035, 0.9604])
```

Over 95% accuracy across all folds - that sounds excellent. But comparing it with a naive baseline exposes how little that really means: a `DummyClassifier` that stubbornly predicts "not a 5" achieves over 90%, simply because only about 10% of all digits are actually 5s.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.90965, 0.90965, 0.90965])
```

**Important:** Why is accuracy misleading for highly imbalanced classes, for example 90% non-5s? Because a classifier that simply always predicts the majority class achieves high accuracy purely because of the class distribution—without having learned anything at all. The naive baseline then almost meets naive expectations for the "real" model. That is why accuracy is usually not the quality metric of choice for imbalanced datasets.

---

## 6 The Confusion Matrix

The confusion matrix counts how often instances of category A were predicted as category B.

**For the binary case ("5" vs. "not 5"), this produces the following schema:**

|                      | **Predicted: Negative** | **Predicted: Positive** |
| -------------------- | ----------------------- | ----------------------- |
| **Actual: Negative** | True Negative (TN)      | False Positive (FP)     |
| **Actual: Positive** | False Negative (FN)     | True Positive (TP)      |

**It is calculated from clean, "out-of-sample" predictions obtained via `cross_val_predict()` (rather than from the test data, which remains untouched):**
```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
# array([[53892,   687],
#        [ 1891,  3530]])

ConfusionMatrixDisplay.from_predictions(y_train_5, y_train_pred)
```

A perfect classifier would have only nonzero values on the main diagonal (TN and TP); all off-diagonal cells (FP, FN) would be empty.

---

## 7 Precision and sensitivity

Two more compact measures can be derived from the confusion matrix:

**Precision ("relevance")** - how many of the cases predicted as positive were actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Sensitivity (recall, "hit rate")** - how many of the actually positive cases were found?

$$\text{Sensitivität} = \frac{TP}{TP + FN}$$

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred)  # 0.8370...
recall_score(y_train_5, y_train_pred)     # 0.6511...
```

This makes the 5-detector look much less impressive than its Accuracy suggested: only 83.7% of its positive predictions are correct and it finds only 65.1% of all actual 5s. If you need a single metric to compare two classifiers, you combine precision and recall into the _F1 score_, the harmonic mean of the two.

**The harmonic mean gives more weight to lower values than the ordinary average does - so a high F1 score can only be achieved if _both_ values are high:**
```python
f1_score(y_train_5, y_train_pred)  # 0.7325...
```

**Important:** Perfect precision can be achieved trivially by having a classifier predict positive only for the one instance it is most confident about and negative for everything else - 100% precision, but practically useless because recall approaches zero. Precision without recall is a metric without substance.

---

## 8 The precision/recall trade-off

The SGDClassifier computes a score for each instance using a decision function; if the score is above a threshold, it is classified as positive. The higher this threshold is, the higher the precision generally is, the lower the recall - and vice versa. So the two metrics cannot be maximized at the same time.

```python
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                              method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

**Which of the two metrics is more important depends entirely on the use case:**

| **Use case**                                   | **Priority**     | **Rationale**                                                                                |
| ---------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------- |
| Spam filter / parental-control filter          | high precision   | better to discard a few legitimate emails/videos than to let something inappropriate through |
| Monitoring system (e.g. detecting shoplifting) | high sensitivity | a few false alarms are acceptable as long as as few real incidents as possible are missed    |

**Note:** Rule of thumb from the book: When someone says, "Let's achieve 99% precision", the follow-up question should always be: "At what sensitivity?". A precision figure without a sensitivity figure is worthless.

---

## 9 The ROC Curve

The ROC curve (Receiver Operating Characteristic) is closely related to the precision-recall curve, but plots the true positive rate (TPR, identical to sensitivity) against the false positive rate (FPR) - the share of negative data points that are incorrectly classified as positive.

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_auc_score(y_train_5, y_scores)  # 0.9604... – 1.0 would be perfect, 0.5 would be coincidence
```

Classifiers can be compared using the _AUC_ (Area Under the Curve): 1.0 for a perfect classifier, 0.5 for a purely random classifier.

| **Curve**              | **When to prefer it**                                                               |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Precision-recall curve | the positive class is rare, or false positives are more costly than false negatives |
| ROC curve              | more balanced class distribution, no strong focus on false positives                |

**Important:** With highly imbalanced classes (such as 5 vs. non-5), the ROC AUC can look deceptively good because there are simply few positives compared with the negatives. In such cases, the PR curve gives a more honest picture of how much room for improvement remains.

---

## 10 Multiclass classification

Multiclass classifiers (multinomial classifiers) distinguish between more than two categories.

**Some algorithms (`LogisticRegression`, `RandomForestClassifier`) can do this directly; others (`SGDClassifier`, `SVC`) are fundamentally binary and require a strategy:**
- _One-versus-Rest (OvR/OvA):_ one binary classifier per category (e.g. 10 for MNIST), prediction = the category with the highest score.
- _One-versus-One (OvO):_ one binary classifier per category pair (N × (N−1) / 2, so 45 for MNIST), prediction = the category that wins the most duels. Advantage: each classifier trains only on the data from the two relevant categories—practical for algorithms such as SVMs, which scale poorly with dataset size.

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])  # Scikit-Learn automatically chooses OvO

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
len(ovr_clf.estimators_)  # 10
```

Scikit-Learn automatically chooses OvO or OvR depending on the algorithm - `SVC` prefers OvO because of poor scaling, while most other binary algorithms (such as `SGDClassifier`) run under the hood using OvR.

---

## 11 Error analysis

**With ten categories instead of two, the confusion matrix quickly becomes hard to read - a color-coded chart helps:**
```python
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred, normalize="true", values_format=".0%"
)
```

Normalizing by row (`normalize="true"`) shows what share of each actual category was classified correctly or incorrectly - important because otherwise categories with many examples visually dominate the matrix without saying anything about model quality. If you additionally set `sample_weight` for the misclassified cases, the actual confusion patterns (e.g. "many digits are incorrectly recognized as 8") stand out much more clearly.

**Note:** Confusion matrices are fundamentally not symmetric: the fact that 10% of all 5s are misclassified as 8s says nothing about how many 8s are incorrectly recognized as 5s (in the book, it's only 2%).

This kind of analysis can point to concrete ways to improve the model: collect more training data for the frequently confused digits, engineer new features (e.g. number of closed loops: an 8 has two, a 6 has one, a 5 has none), or preprocess the images to mitigate known confusion patterns such as shifts and rotations.


---

## 12 Multilabel classification

Sometimes a classifier is expected to output multiple labels at the same time for the same instance - for example in facial recognition ("Alice: yes, Bob: no, Charlie: yes"). In the MNIST example, this can be demonstrated with two artificial labels: "is the digit large (≥7)?" and "is the digit odd?"

```python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train.astype("int8") % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

**For evaluation, the F1 score can be calculated for each label and then averaged. The `average` parameter controls how that averaging is performed:**
```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")     # all labels weighted equally
f1_score(y_multilabel, y_train_knn_pred, average="weighted")  # weighted by Support
```

Classifiers that do not directly support multilabel prediction (e.g. `SVC`) can be linked using `ClassifierChain`: each model in the chain uses the predictions from the previous models in addition to the features, allowing dependencies between labels to be captured (for example that a large digit is more likely to also be odd).

---

## 13 Multioutput classification

The generalization of multilabel classification: each label can now take on more than two possible values. Classic example: removing noise from images - the input is a noisy image of a digit and the output is an array of pixel intensities (one "label" per pixel, with values from 0-255 rather than just True/False).

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

# Denoising-Example:
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train  # Goal: clean original images
y_test_mod = X_test

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
```

**Note:** The line between classification and regression is intentionally blurred here: predicting pixel intensities could just as easily be treated as a regression task. Multioutput systems are also not limited to classification - hybrid forms combining categories and numerical values for each data point are possible as well.

---

## 14 Key terms

| **Term**                       | **Brief Definition**                                                                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Confusion Matrix**           | Table that counts how often instances of an actual category were classified as each predicted category    |
| **Precision**                  | Proportion of cases predicted as positive that are actually positive: TP / (TP + FP)                      |
| **Recall**                     | Proportion of actually positive cases that were correctly identified: TP / (TP + FN)                      |
| **F1 Score**                   | Harmonic mean of precision and recall; high only when both are high                                       |
| **ROC Curve**                  | Plots the true positive rate against the false positive rate for all thresholds                           |
| **AUC**                        | Area under the ROC curve; 1.0 = perfect, 0.5 = random classifier                                          |
| **Multiclass Classification**  | Distinguishing between more than two categories, either directly or via OvR/OvO                           |
| **Multilabel Classification**  | A classifier outputs multiple binary labels per instance                                                  |
| **Multioutput Classification** | Generalization of multilabel classification in which each label can take on more than two values          |
| **Baseline Classifier**        | Naive comparison classifier (e.g., `DummyClassifier`) that shows how much a model is really accomplishing |
| **Data Augmentation**          | Artificially enlarging the training dataset by creating slightly modified copies of existing examples     |

---

## 15 Memorable quotes

1. _Quote:_ High accuracy with imbalanced classes is not an achievement - it's arithmetic.
2. _Quote:_ Precision without recall (sensitivity) is a metric without substance and vice versa.
3. _Quote:_ Anyone who promises "99% precision" without mentioning recall isn't promising much.
4. _Quote:_ The confusion matrix doesn't lie - but it isn't symmetric, either.
5. _Quote:_ A baseline classifier is not a straw man, but the benchmark against which every "real" model must first embarrass itself.
