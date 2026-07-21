# Support Vector Machines

> **Summary:**
>
> Support Vector Machines (SVMs) are supervised learning models for linear and nonlinear classification and regression. Their central idea is geometric: select a decision boundary that leaves the widest possible margin between classes. The observations that define this margin are the **support vectors**. Soft-margin regularization, feature scaling, kernels and carefully selected hyperparameters allow the same principle to handle noisy, nonlinear and continuous-target problems. SVMs are especially effective on small to medium-sized, potentially high-dimensional datasets, but kernel SVMs become expensive as the number of training instances grows.  

**ToC:**
- [1 Why Support Vector Machines matter](#1-why-support-vector-machines-matter)
- [2 SVMs in the QUA³CK process model](#2-svms-in-the-qua³ck-process-model)
- [3 The core idea: the widest possible street](#3-the-core-idea-the-widest-possible-street)
- [4 Decision boundaries, margins, and geometry](#4-decision-boundaries-margins-and-geometry)
- [5 Support vectors](#5-support-vectors)
- [6 Hard-margin and soft-margin classification](#6-hard-margin-and-soft-margin-classification)
- [7 The regularization parameter C](#7-the-regularization-parameter-c)
- [8 Feature scaling](#8-feature-scaling)
- [9 Linear SVM classification in scikit-learn](#9-linear-svm-classification-in-scikit-learn)
- [10 Nonlinear classification with polynomial features](#10-nonlinear-classification-with-polynomial-features)
- [11 Kernels and the kernel trick](#11-kernels-and-the-kernel-trick)
- [12 The polynomial kernel](#12-the-polynomial-kernel)
- [13 The Gaussian RBF kernel and gamma](#13-the-gaussian-rbf-kernel-and-gamma)
- [14 Underfitting and overfitting](#14-underfitting-and-overfitting)
- [15 Hinge loss, squared hinge loss, and optimization](#15-hinge-loss-squared-hinge-loss-and-optimization)
- [16 Support Vector Regression](#16-support-vector-regression)
- [17 Multiclass classification, scores, and probabilities](#17-multiclass-classification-scores-and-probabilities)
- [18 Choosing a scikit-learn implementation](#18-choosing-a-scikit-learn-implementation)
- [19 Computational complexity and scalability](#19-computational-complexity-and-scalability)
- [20 Practical model-selection workflow](#20-practical-model-selection-workflow)
- [21 Common mistakes](#21-common-mistakes)
- [22 Key terms](#22-key-terms)
- [23 Source and validation check](#23-source-and-validation-check)
- [24 Memorable quotes](#24-memorable-quotes)

---

## 1 Why Support Vector Machines matter

**Support Vector Machines are flexible models that can be used for:**
- Linear classification
- Nonlinear classification
- Linear regression
- Nonlinear regression
- Novelty detection and anomaly-related tasks

**They are particularly attractive when:**
- The dataset contains hundreds to a few thousand observations rather than millions.
- The number of features is large, as in text or precomputed image features.
- A relatively clear margin exists between classes.
- A nonlinear boundary is required, but the dataset is still small enough for a kernel model.
- A globally optimal solution is desirable: the standard SVM optimization problem is convex.

SVMs are not usually the first choice for extremely large nonlinear datasets. Kernel-based `SVC` and `SVR` require many pairwise similarity calculations and can scale between approximately quadratic and cubic time in the number of training instances. Linear implementations are considerably more scalable.  

**Important:** An SVM is not automatically nonlinear. The model is linear unless nonlinear features or a nonlinear kernel are introduced.

The material also mentions novelty detection as an SVM application, but treats it as a later topic rather than developing it in this chapter. 

---

## 2 SVMs in the QUA³CK process model

Support Vector Machines fit naturally into the workflow introduced in [docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md](docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md).

| QUA³CK phase                | SVM-related decisions                                                                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Q — Question**            | Determine whether the target is categorical or continuous. Clarify the cost of false positives, false negatives, large regression errors, slow predictions, and slow retraining. |
| **U — Understand the data** | Inspect sample size, feature count, units, scales, outliers, noise, class balance, approximate separability, and missing values.                                                 |
| **A¹ — Acquire**            | Collect representative observations, including difficult cases near expected class boundaries.                                                                                   |
| **A² — Analyze**            | Establish linear baselines, inspect validation errors, and determine whether the data appear to require nonlinear features.                                                      |
| **A³ — Adapt**              | Scale the features, engineer polynomial features where appropriate, select a kernel, and tune `C`, `gamma`, `degree`, `coef0`, or `epsilon`.                                     |
| **C — Communicate**         | Report test-set metrics, confusion matrices, support-vector counts, computational costs, and the consequences of the chosen decision threshold.                                  |
| **K — Keep**                | Store the complete preprocessing-and-model pipeline, hyperparameters, library versions, random seeds, training data definition, and evaluation results.                          |

**Connections to earlier course material:**
- Data quality, units, distributions, and outliers should be examined as described in [docs/course-material/003_Understanding_the_data.md](docs/course-material/003_Understanding_the_data.md).
- Use reproducible environments and pipelines as described in [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md).
- Evaluate classification models using the principles and metrics from [docs/course-material/005_Classification.md](docs/course-material/005_Classification.md).
- Use validation sets, cross-validation, regularization, and bias–variance reasoning from [docs/course-material/006_Training_models.md](docs/course-material/006_Training_models.md).

**Warning:** Scaling, feature generation, and any other learned preprocessing must be fitted only on the training portion of each validation split. A scikit-learn `Pipeline` is the safest way to prevent leakage.

---

## 3 The core idea: the widest possible street

Suppose two classes can be separated by several different lines. A classifier that merely separates the training points may place its boundary very close to one class. Small measurement errors or new observations could then cross the boundary easily.

An SVM instead seeks the separating boundary with the **largest margin**.

**The source material describes this as constructing the _widest possible street_ between the classes:**
- The center line is the decision boundary.
- The edges of the street define the margin.
- The closest observations touch the street boundaries.
- These closest observations are the support vectors.
- Correctly classified observations far away from the street normally do not affect its location.

This is called **large-margin classification**. A wide margin usually indicates that the classifier is less sensitive to small perturbations and may generalize better than an arbitrary separating line.  

The principle should not be interpreted as "maximize the distance between the class centers". SVM training is driven by the observations nearest the boundary, not by class means.

---

## 4. Decision boundaries, margins, and geometry

### 4.1 Linear decision function

For a feature vector (\mathbf{x}), a linear SVM calculates the decision score

[
s(\mathbf{x}) = \mathbf{w}^{T}\mathbf{x} + b
]

where:

* (\mathbf{w}) is the vector of feature weights.
* (b) is the bias or intercept.
* (s(\mathbf{x})) is the signed decision score.

For binary classification:

[
\hat{y} =
\begin{cases}
+1, & s(\mathbf{x}) \geq 0 \
-1, & s(\mathbf{x}) < 0
\end{cases}
]

The decision boundary is the hyperplane

[
\mathbf{w}^{T}\mathbf{x} + b = 0
]

In two dimensions it is a line; in three dimensions it is a plane; in an (n)-dimensional feature space it is an ((n-1))-dimensional hyperplane.

### 4.2 Margin boundaries

SVM notation usually scales the parameters so that the two margin boundaries are

[
\mathbf{w}^{T}\mathbf{x} + b = +1
]

and

[
\mathbf{w}^{T}\mathbf{x} + b = -1
]

The perpendicular distance from an arbitrary point (\mathbf{x}) to the decision boundary is

[
\frac{\left|\mathbf{w}^{T}\mathbf{x} + b\right|}{|\mathbf{w}|}
]

The total width between the two margin boundaries is

[
\frac{2}{|\mathbf{w}|}
]

Therefore, maximizing the margin is equivalent to minimizing (|\mathbf{w}|), usually written as minimizing the smoother objective

[
\frac{1}{2}|\mathbf{w}|^2
]

The source diagrams emphasize this relationship: making the weight vector smaller makes the street wider. 

> **Note:** `decision_function()` returns the signed decision score. For a linear model, it is proportional to geometric distance, but the exact distance requires division by (|\mathbf{w}|).

---

## 5. Support vectors

**Support vectors** are the training observations that determine the final decision boundary.

In a hard-margin solution, they lie on the margin boundaries. In a soft-margin solution, support vectors may:

* Lie on a margin boundary.
* Lie inside the margin.
* Be misclassified.

The important property is that their optimization coefficients are nonzero. Points far from the boundary generally receive zero weight in the dual solution.

For a linear SVM, the weight vector can be recovered from the support vectors:

[
\hat{\mathbf{w}}
================

\sum_{i=1}^{m}
\hat{\alpha}^{(i)}t^{(i)}\mathbf{x}^{(i)}
]

where:

* (t^{(i)} \in {-1,+1}) is the class label.
* (\hat{\alpha}^{(i)}) is the dual coefficient.
* Only observations with (\hat{\alpha}^{(i)} > 0) contribute.

Those observations are the support vectors. 

This gives SVMs a form of sparse representation: prediction depends on selected training observations rather than every training point. However, a difficult or highly overlapping dataset may leave many support vectors, increasing prediction cost.

> **Important:** Adding correctly classified observations far outside the margin will normally leave a stable SVM boundary essentially unchanged. Adding or moving observations near the margin can change it substantially. 

---

## 6. Hard-margin and soft-margin classification

### 6.1 Hard-margin classification

A hard-margin SVM requires every training observation to:

1. Be correctly classified.
2. Lie outside the margin.

For labels (t^{(i)} \in {-1,+1}), the optimization problem is

[
\min_{\mathbf{w},b}
\frac{1}{2}|\mathbf{w}|^2
]

subject to

[
t^{(i)}
\left(
\mathbf{w}^{T}\mathbf{x}^{(i)}+b
\right)
\geq 1,
\qquad i=1,\ldots,m
]

Hard-margin classification has two major weaknesses:

* It works only when the classes are perfectly linearly separable.
* It is highly sensitive to outliers and noise.

A single unusual observation can make the problem infeasible or cause a dramatic shift in the boundary. For this reason, pure hard-margin SVMs are rarely suitable for real data. 

### 6.2 Soft-margin classification

A soft-margin SVM introduces nonnegative slack variables (\xi_i), which quantify margin violations:

[
\xi_i \geq 0
]

The constraints become

[
t^{(i)}
\left(
\mathbf{w}^{T}\mathbf{x}^{(i)}+b
\right)
\geq 1-\xi_i
]

The optimization problem becomes

[
\min_{\mathbf{w},b,\boldsymbol{\xi}}
\left[
\frac{1}{2}|\mathbf{w}|^2
+
C\sum_{i=1}^{m}\xi_i
\right]
]

subject to

[
\xi_i \geq 0
]

This objective balances two goals:

* Keep (|\mathbf{w}|) small, creating a wide margin.
* Penalize observations that enter the margin or cross the decision boundary.

| Property                      | Hard margin                | Soft margin                  |
| ----------------------------- | -------------------------- | ---------------------------- |
| Margin violations             | Forbidden                  | Allowed with a penalty       |
| Requires perfect separability | Yes                        | No                           |
| Outlier sensitivity           | Very high                  | Controlled by regularization |
| Typical real-world use        | Rare                       | Standard                     |
| Main tuning parameter         | None in strict formulation | `C`                          |

---

## 7. The regularization parameter C

The hyperparameter (C) controls how strongly the model penalizes margin violations.

### Low `C`

A small (C):

* Applies stronger regularization.
* Makes violations less expensive.
* Usually permits a wider margin.
* Can make the decision boundary smoother.
* Can reduce overfitting.
* Can cause underfitting if set too low.

### High `C`

A large (C):

* Applies weaker regularization.
* Makes violations expensive.
* Usually produces a narrower margin.
* Tries harder to classify the training observations correctly.
* Can improve fit when the model is too simple.
* Can cause overfitting, especially with noise or outliers.

| Symptom                                                           | Typical `C` adjustment |
| ----------------------------------------------------------------- | ---------------------- |
| Training and validation performance are both poor                 | Increase `C`           |
| Training performance is strong but validation performance is poor | Decrease `C`           |
| Boundary reacts too strongly to individual observations           | Decrease `C`           |
| Margin is too broad and important structure is missed             | Increase `C`           |

The course uses examples such as (C=1) for a broader margin and (C=100) for fewer margin violations. The exact best value is data-dependent and must be selected through validation.  

> **Memory aid:** A high `C` assigns a high **cost** to violations.

---

## 8. Feature scaling

Feature scaling is a fundamental requirement for SVMs.

SVM geometry depends on:

* Dot products
* Euclidean distances
* Norms
* Kernel similarity values

Suppose one feature ranges from (0) to (1), while another ranges from (0) to (100,000). The larger-scale feature can dominate the geometry even when it is not more informative.

Without scaling:

* The margin can become distorted.
* RBF distances may be governed almost entirely by one feature.
* `C` and `gamma` become difficult to interpret.
* Optimization may converge more slowly.
* Model performance may become unstable.

The standard solution is to place a `StandardScaler` inside the model pipeline:

[
x_j^{\prime}
============

\frac{x_j-\mu_j}{\sigma_j}
]

This transforms each feature using its training-set mean and standard deviation.  

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf")
)
```

> **Warning:** Do not scale the full dataset before splitting it. That leaks information from the validation or test data into training.

> **Tip:** Sparse text matrices often require a scaler compatible with sparse input, such as `MaxAbsScaler`, or careful use of `StandardScaler(with_mean=False)`.

---

## 9. Linear SVM classification in scikit-learn

The following example identifies *Iris virginica* using petal length and width.

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = load_iris(as_frame=True)

X = iris.data[
    ["petal length (cm)", "petal width (cm)"]
].to_numpy()

y = iris.target.eq(2).to_numpy()  # True for Iris virginica

svm_clf = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, random_state=42),
)

svm_clf.fit(X, y)
```

Making predictions:

```python
X_new = [
    [5.5, 1.7],
    [5.0, 1.5],
]

predictions = svm_clf.predict(X_new)
print(predictions)
# [ True False]
```

Obtaining decision scores:

```python
scores = svm_clf.decision_function(X_new)
print(scores)
# Approximately [0.66, -0.22]
```

A positive score indicates the positive class, while a negative score indicates the negative class. The magnitude reflects how far the point lies from the decision boundary in decision-score units. 

### 9.1 `LinearSVC` characteristics

`LinearSVC`:

* Is optimized specifically for linear classification.
* Scales well with observations and features.
* Does not use the kernel trick.
* Supports regularization through `C`.
* Uses squared hinge loss by default.
* Does not expose `support_vectors_`, because it uses a different linear solver from `SVC`.
* Does not provide `predict_proba()`.

### 9.2 Linear `SVC` is not the same implementation

This model also produces a linear boundary:

```python
from sklearn.svm import SVC

linear_kernel_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1.0),
)
```

However, `SVC(kernel="linear")` uses the kernel SVM machinery and is generally much slower than `LinearSVC` on large datasets. It does expose support vectors.

> **Tip:** For a genuinely linear problem, start with `LinearSVC` rather than `SVC(kernel="linear")`.

---

## 10. Nonlinear classification with polynomial features

A linear model can create a nonlinear boundary in the original input space if the features are transformed first.

Consider a one-dimensional feature (x_1). Points may not be separable on the original axis. Adding

[
x_2=x_1^2
]

maps each observation to

[
\phi(x_1)=
\begin{pmatrix}
x_1 \
x_1^2
\end{pmatrix}
]

The classes may then become linearly separable in the transformed two-dimensional space.

More generally, `PolynomialFeatures` creates powers and interactions such as:

[
x_1^2,\quad
x_1x_2,\quad
x_2^2,\quad
x_1^3,\ldots
]

A scikit-learn pipeline for the moons dataset is:

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

X, y = make_moons(
    n_samples=100,
    noise=0.15,
    random_state=42,
)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(
        C=10.0,
        max_iter=10_000,
        random_state=42,
    ),
)

polynomial_svm_clf.fit(X, y)
```

This approach is useful when:

* The required interactions are interpretable.
* The original feature count is modest.
* The necessary polynomial degree is low.
* A fast linear estimator is still desirable after transformation.

Its main weakness is feature explosion. With many original features or a high degree, the number of generated terms can become extremely large. 

> **Warning:** Apply polynomial feature generation inside the pipeline so that the complete transformation is reproduced consistently at prediction time.

---

## 11. Kernels and the kernel trick

### 11.1 Why kernels are useful

Explicit nonlinear feature generation can be expensive. A kernel avoids constructing the transformed vectors directly.

Suppose a transformation maps an input into another feature space:

[
\phi:\mathcal{X}\rightarrow\mathcal{H}
]

A valid kernel computes the transformed-space dot product:

[
K(\mathbf{a},\mathbf{b})
========================

\phi(\mathbf{a})^{T}\phi(\mathbf{b})
]

without explicitly calculating (\phi(\mathbf{a})) or (\phi(\mathbf{b})).

This is the **kernel trick**.

### 11.2 Polynomial example

For a second-degree transformation,

[
\phi(\mathbf{x})
================

\begin{pmatrix}
x_1^2 \
\sqrt{2}x_1x_2 \
x_2^2
\end{pmatrix}
]

the transformed dot product can be calculated directly:

[
\phi(\mathbf{a})^{T}\phi(\mathbf{b})
====================================

\left(
\mathbf{a}^{T}\mathbf{b}
\right)^2
]

The transformed vector has three coordinates instead of two, but the kernel evaluates the transformed dot product from the original vectors. 

### 11.3 Why the dual problem matters

The primal linear SVM is expressed in terms of (\mathbf{w}). Its dual formulation is expressed in terms of one coefficient (\alpha_i) per training observation:

[
\max_{\boldsymbol{\alpha}}
\left[
\sum_{i=1}^{m}\alpha_i
----------------------

\frac{1}{2}
\sum_{i=1}^{m}
\sum_{j=1}^{m}
\alpha_i\alpha_j t_i t_j
\mathbf{x}_i^{T}\mathbf{x}_j
\right]
]

subject to constraints such as

[
0\leq \alpha_i \leq C
]

and

[
\sum_{i=1}^{m}\alpha_i t_i=0
]

The inputs appear only through dot products. Therefore, each dot product can be replaced by a kernel:

[
\mathbf{x}_i^{T}\mathbf{x}_j
\quad\longrightarrow\quad
K(\mathbf{x}_i,\mathbf{x}_j)
]

For a new observation, a kernel SVM predicts using

[
f(\mathbf{x})
=============

\sum_{i\in SV}
\alpha_i t_i K(\mathbf{x}_i,\mathbf{x})+b
]

Only support vectors appear because all other (\alpha_i) values are zero.

> **Important:** The kernel trick avoids explicit feature construction, but it does not make computation free. Training still involves a large number of pairwise kernel evaluations.

---

## 12. The polynomial kernel

The polynomial kernel is commonly written as

[
K(\mathbf{a},\mathbf{b})
========================

\left(
\gamma\mathbf{a}^{T}\mathbf{b}+r
\right)^d
]

where:

* (d) is `degree`.
* (r) is `coef0`.
* (\gamma) scales the dot product.

Example:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="poly",
        degree=3,
        coef0=1.0,
        C=5.0,
    ),
)

poly_kernel_svm_clf.fit(X, y)
```

### 12.1 Effect of `degree`

A larger `degree`:

* Adds higher-order interactions.
* Increases model flexibility.
* Can fit more complex boundaries.
* Raises the risk of overfitting.

A smaller `degree`:

* Creates a smoother, less flexible boundary.
* May generalize better on limited or noisy data.
* Can underfit if the real relationship is complex.

### 12.2 Effect of `coef0`

`coef0` controls the relative influence of high-order and low-order polynomial terms.

It is particularly relevant when:

* A high-degree kernel is used.
* The model either ignores useful lower-order structure or gives excessive influence to higher-order terms.

Practical guidance from the source material:

* Reduce `degree` when the model overfits.
* Increase `degree` when it underfits.
* Tune `coef0` rather than treating it as irrelevant. 

> **Tip:** Polynomial kernels can work well when the domain suggests interactions of a particular order. The RBF kernel is often a more convenient first nonlinear baseline when no specific polynomial structure is expected.

---

## 13. The Gaussian RBF kernel and gamma

The Gaussian radial basis function kernel is

[
K(\mathbf{a},\mathbf{b})
========================

\exp
\left(
-\gamma
|\mathbf{a}-\mathbf{b}|^2
\right)
]

It measures similarity:

* (K=1) when the two points are identical.
* The value approaches (0) as their distance increases.

The source material explains this in terms of landmarks. Each landmark produces a similarity feature:

[
\phi(\mathbf{x},\boldsymbol{\ell})
==================================

\exp
\left(
-\gamma
|\mathbf{x}-\boldsymbol{\ell}|^2
\right)
]

If every training observation acts as a landmark, the transformed feature space may be very high-dimensional. The kernel trick obtains the corresponding dot products without constructing all of those features explicitly. 

Example:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

rbf_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        gamma=5.0,
        C=0.001,
    ),
)

rbf_svm_clf.fit(X, y)
```

### 13.1 Meaning of `gamma`

`gamma` controls how quickly similarity falls as distance increases.

#### Low `gamma`

A low value gives each observation a broad radius of influence:

* Similarity decreases slowly.
* Many points influence each local region.
* The decision boundary is smoother.
* Variance is lower.
* Underfitting becomes more likely if `gamma` is too low.

#### High `gamma`

A high value gives each observation a narrow radius of influence:

* Similarity decreases rapidly.
* Influence becomes highly local.
* The boundary can bend around individual observations.
* Model variance increases.
* Overfitting becomes more likely if `gamma` is too high.

| `gamma` | Influence radius | Typical boundary      | Main risk    |
| ------- | ---------------- | --------------------- | ------------ |
| Low     | Broad            | Smooth                | Underfitting |
| High    | Narrow           | Detailed or irregular | Overfitting  |

The sample solution describes increasing `gamma` as making the boundary more "wiggly" because each point influences a smaller neighborhood. 

### 13.2 `gamma="scale"` and `gamma="auto"`

In scikit-learn:

* `"scale"` adapts `gamma` to the feature count and observed feature variance.
* `"auto"` uses a value based on the number of features.

`"scale"` is the default and is usually a sensible starting point, but it does not eliminate the need for scaling or validation.

> **Warning:** `gamma` is distance-dependent. Tuning it on unscaled features is usually misleading.

---

## 14. Underfitting and overfitting

For an RBF SVM, `C` and `gamma` jointly control complexity.

| Situation              |                   `C` |                `gamma` | Expected behavior                             |
| ---------------------- | --------------------: | ---------------------: | --------------------------------------------- |
| Low `C`, low `gamma`   | Strong regularization |        Broad influence | Very smooth; may underfit                     |
| High `C`, low `gamma`  |   Weak regularization |        Broad influence | Smooth but fits errors more aggressively      |
| Low `C`, high `gamma`  | Strong regularization |        Local influence | Local structure with tolerance for violations |
| High `C`, high `gamma` |   Weak regularization | Highly local influence | Very flexible; high overfitting risk          |

### 14.1 When an RBF model underfits

Consider:

* Increasing `C`.
* Increasing `gamma`.
* Improving features.
* Checking whether preprocessing removed useful information.
* Trying a different kernel.

### 14.2 When an RBF model overfits

Consider:

* Decreasing `C`.
* Decreasing `gamma`.
* Improving outlier handling.
* Collecting more representative data.
* Reducing the number of noisy or irrelevant features.

### 14.3 Polynomial model diagnosis

For a polynomial model:

* Increase `degree` when the model is clearly too simple.
* Reduce `degree` when the boundary becomes unnecessarily complex.
* Tune `coef0`.
* Continue tuning `C`, because kernel complexity and regularization are separate dimensions.

> **Important:** Do not change several hyperparameters blindly based on training accuracy alone. Use cross-validation and compare both training and validation performance.

---

## 15. Hinge loss, squared hinge loss, and optimization

### 15.1 Hinge loss

For a label (t\in{-1,+1}) and decision score (s),

[
L_{\text{hinge}}(t,s)
=====================

\max(0,1-ts)
]

Interpretation:

* If (ts\geq1), the observation is correctly classified and outside the margin, so its loss is zero.
* If (0<ts<1), it is correctly classified but inside the margin.
* If (ts<0), it is misclassified.

Hinge loss grows linearly for increasingly serious violations.

### 15.2 Squared hinge loss

Squared hinge loss is

[
L_{\text{squared hinge}}(t,s)
=============================

\left[
\max(0,1-ts)
\right]^2
]

It penalizes large violations quadratically.

| Loss          | Growth after violation | Outlier sensitivity | Typical use in the course material |
| ------------- | ---------------------- | ------------------- | ---------------------------------- |
| Hinge         | Linear                 | Lower               | `SGDClassifier(loss="hinge")`      |
| Squared hinge | Quadratic              | Higher              | Default for `LinearSVC`            |

Squared hinge can put much more pressure on the optimizer to fit extreme observations. This may be useful on clean data but risky when there are substantial outliers.  

### 15.3 Convex optimization

Hard- and soft-margin SVM objectives are convex quadratic optimization problems with linear constraints. Convexity means that a correctly solved SVM objective has a global optimum rather than multiple competing local minima. 

Different implementations use different optimization strategies:

* Kernel `SVC` uses specialized SVM optimization related to the dual problem.
* `LinearSVC` uses algorithms optimized for linear models.
* `SGDClassifier` can optimize hinge loss using stochastic gradient descent.

> **Note:** A global optimum for the selected objective does not guarantee strong test performance. The kernel and hyperparameters may still be inappropriate.

---

## 16. Support Vector Regression

Support Vector Regression reverses the road metaphor.

In classification, the objective is to keep classes **outside** the margin.

In regression, the objective is to place as many observations as possible **inside** an (\epsilon)-wide tolerance tube around the prediction function.

For a predictor (f(\mathbf{x})), the epsilon-insensitive loss is

[
L_{\epsilon}(y,f(\mathbf{x}))
=============================

\max
\left(
0,
|y-f(\mathbf{x})|-\epsilon
\right)
]

Errors smaller than (\epsilon) produce zero loss.

### 16.1 Epsilon-insensitivity

A regression model is called **(\epsilon)-insensitive** because observations inside the tube do not affect the loss.

* Small `epsilon` creates a narrow tube.
* Large `epsilon` creates a broad tube.
* Points outside or on the relevant boundary become support vectors.
* Adding points inside the tube does not normally change the fitted function.

The source material states that reducing (\epsilon) increases the number of support vectors and strengthens the effective regularization, while increasing (\epsilon) reduces the number of support vectors. 

### 16.2 Linear SVR

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

linear_svr = make_pipeline(
    StandardScaler(),
    LinearSVR(
        epsilon=0.5,
        C=1.0,
        random_state=42,
    ),
)

linear_svr.fit(X_train, y_train)
```

### 16.3 Nonlinear SVR

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

polynomial_svr = make_pipeline(
    StandardScaler(),
    SVR(
        kernel="poly",
        degree=2,
        C=0.01,
        epsilon=0.1,
    ),
)

polynomial_svr.fit(X_train, y_train)
```

`LinearSVR` is the regression counterpart of `LinearSVC`, while `SVR` is the regression counterpart of kernel `SVC`. `LinearSVR` scales much better as the number of observations grows; `SVR` can become slow on large training sets. 

### 16.4 Regression hyperparameters

| Hyperparameter | Role                                     |
| -------------- | ---------------------------------------- |
| `epsilon`      | Width of the no-penalty tube             |
| `C`            | Penalty for deviations beyond the tube   |
| `kernel`       | Form of the regression function          |
| `gamma`        | Locality for RBF and some other kernels  |
| `degree`       | Polynomial order for a polynomial kernel |
| `coef0`        | Balance between polynomial term orders   |

> **Warning:** `epsilon` in SVR is not a convergence tolerance. It defines the regression loss tube. Estimator parameters such as `tol` control numerical stopping criteria.

---

## 17. Multiclass classification, scores, and probabilities

### 17.1 Multiclass decomposition

The fundamental SVM formulation is binary. A multiclass problem must therefore be decomposed into multiple binary problems.

The course exercise uses a **one-versus-rest**, also called **one-versus-all**, strategy for the three-class Wine dataset. 

```python
from sklearn.datasets import load_wine
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

wine = load_wine()
X, y = wine.data, wine.target

multiclass_svm = make_pipeline(
    StandardScaler(),
    OneVsRestClassifier(
        LinearSVC(C=1.0, random_state=42)
    ),
)

multiclass_svm.fit(X, y)
```

For (K) classes, one-versus-rest trains (K) binary classifiers. Each classifier separates one class from all others, and the largest decision score determines the prediction.

> **Note:** Some scikit-learn estimators handle multiclass decomposition internally. An explicit wrapper makes the intended strategy clear and reproducible.

### 17.2 Decision scores are not probabilities

`decision_function()` provides confidence-like scores, but these are not calibrated class probabilities.

For a binary model:

* Positive score: positive side of the boundary.
* Negative score: negative side.
* Larger absolute value: farther from the boundary in score units.

Do not interpret a score of `2.0` as an 80%, 90%, or 95% probability.

### 17.3 Probability estimates

`LinearSVC` does not implement `predict_proba()`.

`SVC(probability=True)` can produce probability estimates, but the source explains that this adds an internal calibration process involving cross-validation and logistic fitting. This noticeably increases training cost. 

```python
probabilistic_svc = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42,
    ),
)
```

Use calibrated probabilities only when the application genuinely requires them, such as:

* Cost-sensitive thresholding
* Risk ranking
* Decision support
* Combining predictions from multiple models

---

## 18. Choosing a scikit-learn implementation

| Estimator                     | Task                     |          Kernel support | Scaling | Typical use                                                                         |
| ----------------------------- | ------------------------ | ----------------------: | ------: | ----------------------------------------------------------------------------------- |
| `LinearSVC`                   | Linear classification    |                      No |     Yes | Fast linear baseline; many observations or features                                 |
| `SVC(kernel="linear")`        | Linear classification    | Linear kernel machinery |     Yes | Small datasets where explicit support vectors are useful                            |
| `SVC(kernel="rbf")`           | Nonlinear classification |                     Yes |     Yes | Small to medium nonlinear datasets                                                  |
| `SVC(kernel="poly")`          | Nonlinear classification |                     Yes |     Yes | Problems with plausible polynomial interactions                                     |
| `SGDClassifier(loss="hinge")` | Linear classification    |                      No |     Yes | Very large or out-of-core datasets                                                  |
| `LinearSVR`                   | Linear regression        |                      No |     Yes | Scalable linear SVM regression                                                      |
| `SVR`                         | Nonlinear regression     |                     Yes |     Yes | Small to medium nonlinear regression                                                |
| `OneClassSVM`                 | Novelty detection        |                     Yes |     Yes | Detecting observations unlike the training distribution; not developed further here |

The source offers the following practical rule:

1. Begin with `LinearSVC` when a linear model is plausible or the dataset is large.
2. Move to RBF `SVC` when the linear model is insufficient and the dataset is small enough.
3. Use `SGDClassifier(loss="hinge")` for extremely large or incrementally processed datasets.
4. Use `LinearSVR` or `SVR` according to the same linear-versus-kernel and large-versus-small distinction for regression. 

### 18.1 Decision guide

| Question                                              | Recommended starting point                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| Is the target continuous?                             | `LinearSVR` or `SVR`                                                         |
| Is a linear boundary plausible?                       | `LinearSVC`                                                                  |
| Is the feature space high-dimensional and sparse?     | `LinearSVC` or `SGDClassifier`                                               |
| Does the dataset exceed memory?                       | `SGDClassifier(loss="hinge")` with incremental batches                       |
| Is the dataset small to medium and clearly nonlinear? | RBF `SVC`                                                                    |
| Is there a known polynomial interaction structure?    | Polynomial features plus `LinearSVC`, or polynomial `SVC`                    |
| Are probability estimates mandatory?                  | Calibrated linear model or `SVC(probability=True)`, accepting the extra cost |
| Is nonlinear regression required?                     | `SVR` if the dataset is sufficiently small                                   |

---

## 19. Computational complexity and scalability

Let:

* (m) be the number of training observations.
* (n) be the number of features.

The course material gives the following broad comparison:

| Estimator       |  Approximate training complexity | Out-of-core learning | Kernel trick |
| --------------- | -------------------------------: | -------------------: | -----------: |
| `LinearSVC`     |                          (O(mn)) |                   No |           No |
| `SVC`           |           (O(m^2n)) to (O(m^3n)) |                   No |          Yes |
| `SGDClassifier` | (O(mn)) per effective pass scale |                  Yes |           No |

These expressions are useful rules of thumb, not exact runtime guarantees. Actual cost also depends on:

* Separability
* Tolerance settings
* Feature sparsity
* Kernel parameters
* Number of classes
* Hardware
* Number of support vectors



### 19.1 Kernel matrix cost

A kernel model reasons about many pairs of training points. A full kernel matrix has approximately

[
m^2
]

entries. This becomes a serious memory and computation problem as (m) grows.

For example:

* (m=1,000) gives (1,000,000) pairwise entries.
* (m=10,000) gives (100,000,000) entries.
* (m=100,000) would imply (10^{10}) entries.

This explains why RBF `SVC` may work very well on 2,000 observations but become impractical on 200,000.

### 19.2 Prediction complexity

For a kernel SVM, prediction evaluates the kernel between a new observation and the support vectors:

[
f(\mathbf{x})
=============

\sum_{i\in SV}
\alpha_i t_i K(\mathbf{x}_i,\mathbf{x})+b
]

Therefore, prediction time depends strongly on the number of support vectors. 

A model with almost every training observation acting as a support vector may:

* Require more memory.
* Predict more slowly.
* Indicate substantial overlap, noise, or excessive boundary complexity.

### 19.3 Large nonlinear datasets

When a large dataset needs a nonlinear model, consider:

* Sampling for exploratory kernel experiments.
* Explicit feature approximations followed by a linear estimator.
* Tree ensembles.
* Neural networks.
* Domain-specific feature extraction followed by a linear SVM.

The chapter explicitly recommends considering alternatives such as random forests or neural networks for larger nonlinear tasks. 

---

## 20. Practical model-selection workflow

### Step 1: Split the data correctly

Create a final test set before tuning. Use stratification for classification where appropriate.

### Step 2: Build a leakage-safe pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])
```

### Step 3: Establish a linear baseline

Try `LinearSVC` or `SGDClassifier(loss="hinge")` before introducing kernels.

Questions to ask:

* Does a linear model already perform well?
* Is the dataset high-dimensional?
* Is the model fast enough?
* Are validation errors concentrated in a particular region or class?

### Step 4: Try an RBF kernel when justified

For small to medium data, RBF `SVC` is a strong general nonlinear baseline.

### Step 5: Search `C` and `gamma` logarithmically

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "svc__kernel": ["rbf"],
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [
        "scale",
        "auto",
        0.001,
        0.01,
        0.1,
        1,
    ],
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    refit=True,
)

search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)
```

The source recommends:

1. A broad initial search.
2. A finer search around promising values.
3. Five-fold cross-validation for robust estimates.
4. Three-fold cross-validation when computation on a very large dataset is restrictive.
5. Final evaluation on the untouched test set. 

### Step 6: Select metrics that match the question

For classification, consider:

* Accuracy
* Balanced accuracy
* Precision
* Recall
* (F_1)
* ROC AUC or PR AUC
* Confusion matrix
* Class-specific error costs

For regression, consider:

* MAE
* RMSE
* (R^2)
* Error distributions
* Performance across important target ranges

### Step 7: Inspect complexity as well as score

Record:

* Training time
* Prediction time
* Number of support vectors
* Cross-validation variability
* Memory requirements
* Calibration cost, when probabilities are enabled

### Step 8: Refit and evaluate once

After model selection:

1. Refit the complete pipeline on the available training data.
2. Evaluate once on the test set.
3. Save the complete pipeline.
4. Document all preprocessing and hyperparameters.

> **Tip:** `RandomizedSearchCV` is often more efficient than exhaustive grid search when several kernel parameters or broad continuous ranges must be explored.

---

## 21. Common mistakes

### 1. Training an SVM without scaling

A feature with a large numeric range can dominate the margin and RBF distance calculations.

**Correction:** Use a scaler inside a pipeline.

### 2. Scaling before the train–test split

This leaks information from the test set.

**Correction:** Fit preprocessing only inside the training pipeline.

### 3. Treating high `C` as stronger regularization

The relationship is the opposite in standard SVM APIs.

**Correction:** Lower `C` means stronger regularization and greater tolerance for violations.

### 4. Assuming `gamma` is another name for regularization

`gamma` controls locality in kernels such as RBF. It does not replace `C`.

**Correction:** Tune `C` and `gamma` jointly.

### 5. Using kernel `SVC` on an enormous dataset

Kernel training can become quadratically or cubically expensive in (m).

**Correction:** Start with `LinearSVC`, `SGDClassifier`, sampling, or another scalable model.

### 6. Interpreting `decision_function()` as a probability

A decision score is not calibrated to ([0,1]).

**Correction:** Use explicit calibration only when probabilities are required.

### 7. Enabling `probability=True` by default

Probability calibration adds substantial training cost.

**Correction:** Enable it only for a defined application need.

### 8. Selecting the model from training accuracy

A large `C`, high `gamma`, or high polynomial degree may memorize noise.

**Correction:** Compare training and cross-validation performance.

### 9. Using a high polynomial degree without checking feature growth

Explicit polynomial features may expand combinatorially.

**Correction:** Monitor transformed dimensionality or use a kernel.

### 10. Confusing `epsilon` with `tol`

In SVR, `epsilon` defines the insensitive tube. `tol` is an optimization stopping tolerance.

### 11. Assuming every training observation contributes equally

Kernel predictions are driven primarily by support vectors.

**Correction:** Inspect `support_`, `support_vectors_`, and `n_support_` when using `SVC` or `SVR`.

### 12. Assuming support vectors are always correctly classified points on the margin

In soft-margin SVMs, support vectors may also lie inside the margin or on the wrong side of the decision boundary.

### 13. Ignoring outliers when using squared hinge loss

Squared hinge penalizes extreme violations quadratically.

**Correction:** Clean the data, tune regularization, or consider hinge loss through `SGDClassifier`.

### 14. Applying a nonlinear kernel before establishing a baseline

A complex model may improve training performance while adding unnecessary cost.

**Correction:** Begin with a scaled linear baseline.

### 15. Tuning on the test set

Repeated test-set use turns the test set into another validation set.

**Correction:** Reserve it for one final evaluation.

---

## 22. Key terms

| Term                            | Meaning                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Support Vector Machine**      | A large-margin model for classification, regression, and related tasks.                          |
| **Hyperplane**                  | An ((n-1))-dimensional decision boundary in an (n)-dimensional feature space.                    |
| **Margin**                      | The distance between the decision boundary and the nearest relevant observations.                |
| **Large-margin classification** | Selecting a boundary that maximizes the margin.                                                  |
| **Support vector**              | A training point with a nonzero dual coefficient that helps determine the boundary.              |
| **Hard margin**                 | SVM formulation that permits no violations and requires linear separability.                     |
| **Soft margin**                 | SVM formulation that allows violations and penalizes them.                                       |
| **Slack variable (\xi_i)**      | A nonnegative quantity measuring the degree of a margin violation.                               |
| **`C`**                         | Hyperparameter controlling the penalty assigned to violations.                                   |
| **Decision score**              | The value (s(\mathbf{x})=\mathbf{w}^{T}\mathbf{x}+b).                                            |
| **Hinge loss**                  | (\max(0,1-ts)); linear penalty for margin violations.                                            |
| **Squared hinge loss**          | ([\max(0,1-ts)]^2); quadratic penalty for violations.                                            |
| **Kernel**                      | A function computing an inner product in a transformed feature space.                            |
| **Kernel trick**                | Using transformed-space dot products without explicitly constructing transformed features.       |
| **Polynomial kernel**           | Kernel of the form ((\gamma\mathbf{a}^{T}\mathbf{b}+r)^d).                                       |
| **RBF kernel**                  | Gaussian similarity kernel (\exp(-\gamma|\mathbf{a}-\mathbf{b}|^2)).                             |
| **`gamma`**                     | Hyperparameter controlling the influence radius of observations in the RBF kernel.               |
| **`degree`**                    | Polynomial order used by the polynomial kernel.                                                  |
| **`coef0`**                     | Offset controlling the balance among polynomial term orders.                                     |
| **Primal problem**              | SVM optimization expressed directly in terms of (\mathbf{w}), (b), and possibly slack variables. |
| **Dual problem**                | Equivalent optimization expressed through coefficients attached to training observations.        |
| **Convex optimization**         | Optimization with no competing local optima; a global optimum exists for the SVM objective.      |
| **SVR**                         | Support Vector Regression, which fits an (\epsilon)-insensitive tube around predictions.         |
| **Epsilon-insensitive loss**    | Loss that ignores absolute regression errors no greater than (\epsilon).                         |
| **One-versus-rest**             | Multiclass strategy that trains one binary classifier per class.                                 |
| **Out-of-core learning**        | Training incrementally on data that do not fit in memory.                                        |
| **Underfitting**                | A model is too simple to capture the relevant structure.                                         |
| **Overfitting**                 | A model follows training-specific noise and generalizes poorly.                                  |

---

## 23. Source and validation check

The summary consolidates the supplied material rather than translating any single document literally.

Reviewed sources:

* `2 - Poster.png`
* `3 - Folien - Support_Vector_Machines_Das_Bauen_der_optimalen_Stra├ƒe.pdf` — all 15 visually rendered pages reviewed. 
* `4 - Pra╠êsentation - Support-Vector-Machines.pdf` — classification, kernels, regression, optimization, implementation comparison, tuning, exercises, and glossary reviewed. 
* `5 - Musterloesung Large-margin Geometry.pdf` — all supplied questions, explanations, deep dives, and terminology reviewed. 
* `7 - Machine Learning Kapitel 5.pdf` — complete supplied chapter extract reviewed. 

The structure was aligned with the existing course summaries. The Python examples use valid scikit-learn APIs, the mathematical expressions use valid LaTeX, repeated material was consolidated, and the prose was normalized into consistent technical English.

---

## 24. Memorable quotes

> **"An SVM does not merely draw a separating line; it builds the widest possible street between the classes."**

> **"The boundary is supported by the observations closest to it, not by the observations farthest away."**

> **"Low `C` tolerates mistakes to protect the margin; high `C` protects the training points at the expense of the margin."**

> **"`gamma` controls how local the model’s attention becomes."**

> **"Scale first, then fit."**

> **"For regression, the road is no longer kept empty—the goal is to place as many observations as possible inside it."**

> **"Start linear. Add nonlinear complexity only when validation evidence justifies it."**
