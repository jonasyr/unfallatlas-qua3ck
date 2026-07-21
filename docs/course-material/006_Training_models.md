# Training models

> **Summary:**
>
> [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md) and [docs/course-material/005_Classification.md](docs/course-material/005_Classification.md) have so far treated models as a black box: you call `.fit()` and `.predict()` and evaluate the result without knowing what happens in between. This file opens that very box - using linear regression as an example. **The through line:** Training a model means minimizing a cost function - either in a single step using a closed-form equation (the normal equation) or iteratively using a gradient-based method. With greater model complexity (polynomial regression), the overfitting problem from [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md) `## 12 The biggest challenges: poor algorithms` returns in full force - and here it is diagnosed with learning curves and addressed with regularization (ridge, lasso, elastic net, early stopping). Finally, the same toolkit is applied to classification: logistic and softmax regression.

**ToC:**
- [1 Open the black box](#1-open-the-black-box)
- [2 Classification within the QUA³CK model](#2-classification-within-the-qua³ck-model)
- [3 Linear regression: Basic idea and cost function](#3-linear-regression-basic-idea-and-cost-function)
- [4 The normal equation](#4-the-normal-equation)
- [5 Gradient Descent: Basic principle](#5-gradient-descent-basic-principle)
- [6 Batch gradient descent](#6-batch-gradient-descent)
- [7 Stochastic gradient descent](#7-stochastic-gradient-descent)
- [8 Mini-batch gradient descent and SGD with scikit-learn](#8-mini-batch-gradient-descent-and-sgd-with-scikit-learn)
- [9 Polynomial regression](#9-polynomial-regression)
- [10 Learning curves](#10-learning-curves)
- [11 Regularized linear models: Ridge regression](#11-regularized-linear-models-ridge-regression)
- [12 Lasso regression](#12-lasso-regression)
- [13 Elastic Net](#13-elastic-net)
- [14 Early Stopping](#14-early-stopping)
- [15 Logistic regression](#15-logistic-regression)
- [16 Decision boundaries](#16-decision-boundaries)
- [17 Softmax regression](#17-softmax-regression)
- [18 Common mistakes](#18-common-mistakes)
- [19 Key terms](#19-key-terms)

---

## 1 Open the black box

Training a model means adjusting its parameters so that it represents the training data as well as possible. So far, calling `.fit()` has been enough. This unit asks what `.fit()` actually computes - using what is probably the simplest model of all: linear regression.

**There are two fundamentally different training strategies to choose from:**
- a _closed-form equation_ that directly produces the optimal parameters in a single computational step (the normal equation)
- an _iterative optimization method_, gradient descent (GD), which adjusts the parameters step by step to minimize a cost function

**Note:** Why this is worthwhile: A basic understanding of training methods helps you quickly find a suitable model along with the right training procedure and hyperparameters, makes troubleshooting easier - and is the foundation for everything that later leads to neural networks.

Starting from linear regression, the path leads through polynomial regression - which can represent nonlinear data using a linear model, but is more prone to overfitting - to regularization techniques, and finally to two models for classification tasks: logistic regression and softmax regression.

---

## 2 Classification within the QUA³CK model

| **Phase** | **Meaning**                                                       | **Role in This File**                                                                                                                                                                                                                                                                  |
| --------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Q**     | Question                                                          | not directly affected                                                                                                                                                                                                                                                                  |
| **U**     | Understanding the Data                                            | Feature scaling is a prerequisite for a functioning gradient-based method                                                                                                                                                                                                              |
| **A³**    | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | the deepest dive in the series into _Adjusting Hyperparameters_: `eta` (learning rate), `alpha` (regularization strength) and `degree` (polynomial degree) are explained mechanistically here for the first time, rather than simply optimized through grid search                     |
| **C**     | Conclude & Compare                                                | Learning curves provide an additional diagnostic tool alongside the metrics from [docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md](docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md) `## 8 C - Conclude & Compare: Evaluate models` |
| **K**     | Knowledge Transfer                                                | not directly affected                                                                                                                                                                                                                                                                  |

**Note:** This file is the mechanistic continuation of [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md) `## 12 The biggest challenges: poor algorithms`: There, overfitting and underfitting were described only as symptoms with generic countermeasures ("simplify the model", "loosen restrictions"). Here, those countermeasures are given concrete hyperparameter names.

**The hyperparameters in this unit can be mapped directly to the error patterns from [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md) `## 12 The biggest challenges: poor algorithms`:**

| **Error Pattern**             | **Hyperparameter in This File**            | **Effect**                                                    |
| ----------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| Overfitting                   | increase `alpha` (Ridge/Lasso/Elastic Net) | stronger regularization, smaller weights                      |
| Overfitting                   | enable early stopping                      | training ends before the validation error starts rising again |
| Overfitting                   | lower `degree` in `PolynomialFeatures`     | less model capacity                                           |
| Underfitting                  | increase `degree` in `PolynomialFeatures`  | more model capacity                                           |
| Underfitting                  | lower `alpha`                              | weaker regularization, more freedom for the weights           |
| Divergence/long training time | adjust `eta` (learning rate)               | too large → divergence, too small → slow convergence          |

---

## 3 Linear regression: Basic idea and cost function

**A linear model makes its prediction as a weighted sum of the input features plus a constant, the bias term:**
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$$

In vector notation: $\hat{y} = h_\theta(x) = \theta \cdot x$, where $\theta$ is the model's parameter vector (including the bias $\theta_0$) and $x$ is the feature vector of a data point (with $x_0 = 1$).

**To train this model, it needs a performance criterion. In practice, instead of RMSE, the _mean squared error (MSE)_ is usually used as the cost function - both are minimized at the same point, but MSE is easier to optimize:**
$$\text{MSE}(\theta) = \frac{1}{m}\sum_{i=1}^{m}\left(\theta^\top x^{(i)} - y^{(i)}\right)^2$$

Training therefore means finding the value of $\theta$ that minimizes this cost function over the training data.

**Note:** The training and evaluation metrics do not have to be identical. Classifiers are often trained using an easily optimizable cost function (e.g. log loss; see `## 15 Logistic regression`), but evaluated using precision/recall - as long as the two quantities are strongly correlated, this is not a contradiction.

---

## 4 The normal equation

**For the MSE of a linear regression, there is a closed-form solution that computes $\theta$ directly - the _normal equation_:**
$$\hat{\theta} = (X^\top X)^{-1} X^\top y$$

```python
import numpy as np
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)
m = 100  # Number of instances
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = add_dummy_feature(X)  # Add bias column (x0 = 1)

theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("Normalengleichung theta:", theta_best.ravel())
# Expected result: approx. [4, 3]
```

The data was artificially generated from $y = 4 + 3x_1 + \text{noise}$ - the normal equation finds $\theta$ values close to $[4, 3]$, but not exactly, because the noise makes a perfect reconstruction impossible.

**Warning:** _Computational complexity:_ The normal equation inverts an $(n+1) \times (n+1)$ matrix, which costs about $O(n^{2.4})$ to $O(n^3)$ computation time ($n$ = number of features). Scikit-Learn's SVD-based approach, which `LinearRegression` actually uses, runs in $O(n^2)$—better, but with a very large number of features, such as 100,000, both methods become impractically slow. In terms of the number of training instances $m$, however, both are linear ($O(m)$) and can therefore handle large datasets without problems, as long as they fit in memory. With a very large number of features, a gradient-based method is the better choice.

---

## 5 Gradient Descent: Basic principle

_Gradient descent (GD)_ is a general optimization algorithm: it iteratively changes the parameters in the direction in which the cost function decreases most steeply - like descending a mountain in thick fog, where you can only feel the slope beneath your feet. Once the gradient becomes zero, a minimum has been reached.

The central hyperparameter is the _learning rate_ `eta` ($\eta$): it determines the step size.

**Important:** If the learning rate is too small, convergence takes a very long time. If it is too large, the algorithm can jump past the minimum and _diverge_ - the cost function then increases from one step to the next instead of decreasing.

As the cost function for linear regression, MSE is convex and continuously differentiable - so there is only one global minimum and no risk of local minima. This is not true for every cost function: with more irregular landscapes, a random initialization can get stuck in a local minimum or on a long plateau.

**Warning:** If features are scaled differently, the cost function becomes an elongated bowl rather than a round one, and gradient descent takes much longer to reach the minimum. Always scale features before using gradient descent (e.g. with `StandardScaler`).

---

## 6 Batch gradient descent

**In batch gradient descent, the gradient is computed over the _entire_ training dataset at each step:**
$$\nabla_\theta \text{MSE}(\theta) = \frac{2}{m} X^\top (X\theta - y)$$

```python
# X_b, m, y taken from ## 4 The normal equation
eta = 0.1
n_epochs = 1000

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print("Batch-GD theta:", theta.ravel())
```

Each complete pass through the training data is called an _epoch ("Epoche")_. With a suitable learning rate, this code produces exactly the same result as the Normal Equation in `## 4 The normal equation` after 1,000 epochs.

**The downside:** at every single step, the entire dataset is processed, which makes the method slow for large training datasets - but it scales well with the number of features.

---

## 7 Stochastic gradient descent

_Stochastic gradient descent (SGD)_ goes to the other extreme: at each step, it selects only _one random data point_ and computes the gradient only for that point. This makes each step much faster and enables training on huge datasets (including out-of-core training), but the path to the minimum becomes much more irregular - the cost function bounces around instead of decreasing smoothly, and it never fully settles down at the minimum.

**The usual solution:** gradually reduce the learning rate using a _learning schedule_ (simulated annealing) - large steps at the beginning, smaller ones toward the end.

```python
# X_b, m, y taken from ## 4 The normal equation
t0, t1 = 5, 50  # Hyperparameters for the learning schedule

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
        gradients = 2 * xi.T @ (xi @ theta - yi)  # for SGD, do not divide by m
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

print("SGD theta:", theta.ravel())
```

With only 50 instead of 1,000 epochs, SGD already reaches a good solution. The randomness also helps it escape irregular cost functions when they get stuck in a local minimum—an advantage over the batch method once cost functions are no longer convex.

**Note:** For SGD to truly move toward the global optimum on average, the training instances must be independent and identically distributed (for example by shuffling). If they are sorted by label, SGD optimizes one label at a time and misses the global minimum.

---

## 8 Mini-batch gradient descent and SGD with scikit-learn

_Mini-batch gradient descent_ falls between the two extremes: it computes gradients on small, random subsets (mini-batches) instead of on the entire dataset or just one point. The main advantage over SGD is that it can take advantage of hardware optimized for matrix operations, especially GPUs.

**In practice, scikit-learn's `SGDRegressor` handles this work:**
```python
# X, y taken from ## 4 The normal equation
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None,
                       eta0=0.01, n_iter_no_change=100,
                       random_state=42)
sgd_reg.fit(X, y.ravel())
print("SGDRegressor intercept:", sgd_reg.intercept_, "coef:", sgd_reg.coef_)
```

`SGDRegressor` trains for either 1,000 epochs (`max_iter`) or stops earlier if the loss improves by less than `tol` over 100 epochs (`n_iter_no_change`).

| **Method**        | **Convergence behavior**                                      | **Computational cost per step** | **Suitability for large datasets**            |
| ----------------- | ------------------------------------------------------------- | ------------------------------- | --------------------------------------------- |
| **Batch GD**      | smooth, ends exactly at the minimum                           | high (entire dataset)           | poor (slow for large $m$), good for large $n$ |
| **Stochastic GD** | irregular, oscillates around the minimum                      | very low (1 data point)         | very good, supports out-of-core learning      |
| **Mini-batch GD** | smoother than SGD, oscillates more tightly around the minimum | low (small subset)              | good, also GPU-friendly                       |

**Tip:** After training, there is hardly any difference between the methods: all three produce very similar models. The choice is therefore primarily a matter of training speed and dataset size, not later model quality.

---

## 9 Polynomial regression

Nonlinear data can also be fit with a linear model: you simply add powers of each feature as additional features and train a linear model on the expanded feature set.

**Scikit-Learn's `PolynomialFeatures` handles this transformation:**
```python
PolynomialFeatures(degree=90, include_bias=False)
```

The `degree` hyperparameter directly controls model capacity: `PolynomialFeatures(degree=d)` generates $\binom{n+d}{d}$ new features from $n$ features, including all combinations when there are multiple original features. With a high `degree` and several features, the number of features can grow combinatorially.

**Warning:** A high polynomial degree fits the training data more and more closely, but it also pushes the model straight into overfitting. A degree that is too low (for example `degree=1`, meaning ordinary linear regression) leads to underfitting on nonlinear data. This makes `degree` a classic overfitting/underfitting lever in the sense of [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md) `## 12 The biggest challenges: poor algorithms`.

---

## 10 Learning curves

To diagnose whether a model is overfitting or underfitting, a second tool can help alongside cross-validation: _learning curves_. They plot training and validation error against the size of the training dataset.

```python
# X, y taken from ## 4 The normal equation
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y,
    train_sizes=np.linspace(0.01, 1.0, 40),
    cv=5,
    scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)
print("Learning curve - final training error:", train_errors[-1].round(4),
      "| Validation error:", valid_errors[-1].round(4))
```

**The shape of the curves directly reveals which problem from [docs/course-material/004_The_machine_learning_environment.md](docs/course-material/004_The_machine_learning_environment.md) `## 12 The biggest challenges: poor algorithms` is present:**

| **Curve pattern**                                                                                  | **Diagnosis** | **Typical cause**                                                               |
| -------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------- |
| both curves approach a plateau, are close together, and _high_                                     | Underfitting  | Model too simple for the data structure                                         |
| training error remains _low_, validation error remains well above it, large gap between the curves | Overfitting   | Model too complex (e.g., high polynomial degree) relative to the amount of data |

**Tip:** With underfitting, more training data won't help - what's needed is a more powerful model or better features. With overfitting, more training data can help until the curves move closer together; alternatively, use regularization (`## 11`-`## 14`) or a simpler model.

---

## 11 Regularized linear models: Ridge regression

Regularization restricts a model's degrees of freedom to make overfitting more difficult - usually, in linear models, through a constraint on the weights. _Ridge regression_ (Tikhonov regularization) adds a penalty term to the MSE proportional to the squared $\ell_2$ norm of the weight vector:

$$J(\theta) = \text{MSE}(\theta) + \frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2$$

The bias term $\theta_0$ remains untouched (the sum starts at $i=1$). The hyperparameter `alpha` controls the strength: when $\alpha = 0$, Ridge is equivalent to ordinary linear regression; when $\alpha$ is very large, all weights become nearly zero.

```python
# X, y taken from ## 4 The normal equation
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

**Warning:** Ridge regression is sensitive to the scale of the input features—just like with gradient-based methods: _scale first_ (e.g. `StandardScaler`), _then_ regularize, not the other way around.

---

## 12 Lasso regression

The _Lasso regression_ (Least Absolute Shrinkage and Selection Operator) uses the $\ell_1$ norm of the weight vector as the penalty term instead of the $\ell_2$ norm:

$$J(\theta) = \text{MSE}(\theta) + 2\alpha\sum_{i=1}^{n}|\theta_i|$$

```python
# X, y taken from ## 4 The normal equation
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```

**The key difference from ridge:** lasso tends to set the weights of unimportant features completely to zero, rather than merely shrinking them. This is both a side effect and a feature: lasso thereby implicitly performs _automatic feature selection_ and produces a sparse model with only a few nonzero weights.

---

## 13 Elastic Net

_Elastic Net_ combines Ridge and Lasso through a mixing parameter $r$ (`l1_ratio` in scikit-learn):

$$J(\theta) = \text{MSE}(\theta) + r \cdot 2\alpha\sum_{i=1}^{n}|\theta_i| + (1-r)\frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2$$

When $r=0$, Elastic Net is equivalent to pure Ridge regression; when $r=1$, it is equivalent to pure Lasso.

```python
# X, y taken from ## 4 The normal equation
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

As a rule of thumb, plain linear regression (with no regularization at all) should be avoided - a little regularization is almost always better. Ridge is a solid default starting point. If only a few features are believed to be truly relevant, Lasso or Elastic Net is preferable. Elastic Net is generally preferred over pure Lasso because Lasso can become unstable when there are more features than training instances or when features are highly correlated.

---

## 14 Early Stopping

A completely different regularization approach for iterative methods such as gradient descent is to stop training as soon as the validation error has reached its minimum - _Early Stopping_. If training continues, the training error may keep decreasing, but the validation error begins to rise again: the model is overfitting.

```python
# X, y taken from ## 4 The normal equation
from copy import deepcopy
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

preprocessing = make_pipeline(
    PolynomialFeatures(degree=90, include_bias=False),  # like in ## 9 Polynomial regression
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

print(f"Early Stopping – best Validation error: {best_valid_rmse:.4f}")
```

Instead of `fit()`, this uses `partial_fit()` to train the model incrementally, epoch by epoch and measure its performance on the validation set along the way.

**Tip:** The code does not actually stop training—it continuously keeps a copy of the best model so far using `deepcopy(sgd_reg)`. That is no accident: Geoffrey Hinton called Early Stopping a “beautiful free lunch,” but the champagne belongs to the _best_ model, not the _last_ one. Anyone who uses `sgd_reg` itself after the loop may be celebrating a model that has already started overfitting again.

---

## 15 Logistic regression

_Logistic regression_ estimates the probability that a data point belongs to a particular category, making it a classification method despite its name. Instead of outputting the result of the weighted sum directly, as linear regression does, it passes that result through the logistic (sigmoid) function:

$$\hat{p} = h_\theta(x) = \sigma(\theta^\top x), \qquad \sigma(t) = \frac{1}{1+\exp(-t)}$$

The result lies between 0 and 1. The model is trained using _log loss_ (cross-entropy in the binary case) as the cost function; unlike linear regression, there is no closed-form solution for it, but the function is convex, so a gradient-based method is guaranteed to converge to the global optimum.

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
print("Logistic regression - probability for petal_width=1.5:",
      log_reg.predict_proba([[1.5]]).round(3))
```

`predict_proba()` returns the estimated probabilities for both categories, while `predict()` returns the resulting hard prediction.

---

## 16 Decision boundaries

If the estimated probability $\hat p \geq 0.5$, the model predicts the positive category; otherwise, it predicts the negative one - the default threshold is therefore 50%. In the Iris example (petal width), the probabilities for the two categories move continuously in opposite directions; the point where the two curves intersect at 50% is the model's _decision boundary_ (in the book example, at about 1.65 cm petal width).

```python
# log_reg taken from ## 15 Logistic regression
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]
```

With more than one feature, the decision boundary becomes a line (or, with even more features, a hyperplane) in the feature space. Two categories with overlapping feature ranges inevitably create an uncertainty region around the boundary, where the model is not confident but still has to predict a category.

**Note:** The regularization hyperparameter for logistic regression in scikit-learn is not called `alpha`, but `C`—its _reciprocal_. The larger `C` is, the weaker the regularization (the opposite of `alpha` in Ridge/Lasso/Elastic Net).

---

## 17 Softmax regression

_Softmax regression_ (multinomial logistic regression) directly generalizes logistic regression to more than two categories, without having to combine multiple binary classifiers (unlike OvR/OvO from [docs/course-material/005_Classification.md](docs/course-material/005_Classification.md) `## 10 Multiclass classification`). For each category $k$, a separate score is calculated and then converted into a probability using the softmax function:

$$\hat p_k = \frac{\exp(s_k(x))}{\sum_{j=1}^{K}\exp(s_j(x))}$$

The category with the highest probability (`argmax`) is predicted. Training uses _cross-entropy_ as the cost function, the generalized form of log loss for more than two categories.

```python
# iris, LogisticRegression, train_test_split taken from ## 15 Logistic regression
X_sm = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_sm = iris["target"]

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(
    X_sm, y_sm, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train_sm, y_train_sm)
print("Softmax - Score:", round(softmax_reg.score(X_test_sm, y_test_sm), 3))
print("Softmax - Prediction for [5, 2]:", softmax_reg.predict([[5, 2]]))
print("Softmax - Probabilities:", softmax_reg.predict_proba([[5, 2]]).round(2))
```

Scikit-Learn's `LogisticRegression` automatically uses softmax regression internally as soon as more than two categories are present in the training dataset (with the default solver, `lbfgs`) - there is no separate "SoftmaxRegression" classifier.

**Important:** Softmax regression always predicts only _one_ category per data point (the categories are mutually exclusive). It is not suitable for tasks where multiple categories can apply at the same time (multilabel; see [docs/course-material/005_Classification.md](docs/course-material/005_Classification.md) `## 12 Multilabel classification`).

---

## 18 Common mistakes

| **Problem**                                                               | **Better**                                                                                                                                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Learning rate set too high → cost function diverges instead of decreasing | Gradually reduce the learning rate (learning schedule) or try smaller fixed values as a test                                                                       |
| Features not scaled before using gradient descent                         | Always apply `StandardScaler` (or something similar) before training—this applies equally to GD-based methods and to Ridge/Lasso/Elastic Net                       |
| Early stopping without a copy of the best model (`deepcopy`)              | Save a `deepcopy` of the model in each epoch when it improves—otherwise, you are only keeping the _last_ model, not the _best_ one                                 |
| Regularization applied before scaling                                     | Scale first, then regularize—regularization on unscaled data systematically penalizes features with large value ranges more heavily                                |
| Blindly increasing the polynomial degree to lower the training error      | Check learning curves instead of only watching the training error—a falling training error with a growing gap to the validation error is overfitting, not progress |

---

## 19 Key terms

| **Term**                        | **Brief Definition**                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Cost function**               | A measure of how poorly a model currently represents the training data; minimized during training                                        |
| **Normal equation**             | A closed-form equation that directly calculates the optimal parameters of a linear regression model                                      |
| **Gradient descent**            | An iterative optimization method that gradually adjusts parameters in the direction of a decreasing cost function                        |
| **Learning rate**               | A hyperparameter (`eta`) that controls the step size in gradient descent                                                                 |
| **Batch gradient descent**      | Calculates the gradient at each step using the entire training dataset                                                                   |
| **Stochastic gradient descent** | Calculates the gradient at each step using only one random data point                                                                    |
| **Mini-batch gradient descent** | Calculates the gradient using small, random subsets of the training data                                                                 |
| **Learning curve**              | A plot of training and validation error over the size of the training dataset; used to diagnose overfitting and underfitting             |
| **Ridge regression**            | Linear regression with an $\ell_2$ regularization term ($\alpha$ controls the strength)                                                  |
| **Lasso regression**            | Linear regression with an $\ell_1$ regularization term; sets the weights of unimportant features to zero (feature-selection side effect) |
| **Elastic Net**                 | A hybrid of Ridge and Lasso, controlled by the mixing parameter `l1_ratio`                                                               |
| **Early stopping**              | Regularization by stopping training once the validation error reaches its minimum                                                        |
| **Logistic regression**         | A classification model that estimates probabilities for a binary category using the sigmoid function                                     |
| **Softmax regression**          | A generalization of logistic regression to more than two mutually exclusive categories                                                   |
| **Decision boundary**           | The point or surface in feature space where a classifier switches between categories (default: $\hat p = 0.5$)                           |

---

## 20 Memorable quotes

1. _Quote:_ Training means minimizing a cost function - either in one step (normal equation) or in many small ones (gradient descent).
2. _Quote:_ Scaling is not a side issue: without it, the cost-function bowl turns into a valley that gradient descent first has to make its way through.
3. _Quote:_ A decreasing training error is not progress as long as the gap to the validation error is growing.
4. _Quote:_ Regularization is almost never the wrong choice - plain linear regression without any constraint is the exception, not the norm.
5. _Quote:_ Early stopping only remembers the best model if you explicitly copy it - otherwise, the last one may randomly end up winning.
