# Understanding the data - data exploration and data preparation

> **Summary:**
>
> This file explores the _U phase_ of the QUA³CK model in greater depth: _Understanding the Data_. The focus is on how raw data is systematically examined, cleaned, visualized and prepared before the actual modeling begins. It is precisely at this stage that it is often determined whether a machine learning project will later yield useful results or simply fail spectacularly, like so many of humanity's promising digital projects.

**ToC:**
- [1 Why is "Understanding the Data" so important?](#1-why-is--understanding-the-data-so-important)
- [2 Classification within the QUA³CK model](#2-classification-within-the-qua³ck-model)
- [3 Data sources in machine learning](#3-data-sources-in-machine-learning)
- [4 Understanding data types](#4-understanding-data-types)
    - [Structured data](#structured-data)
    - [Semi-structured data](#semi-structured-data)
    - [Unstructured data](#unstructured-data)
- [5 The goal of Data Exploration](#5-the-goal-of-data-exploration)
- [6 Basic Analysis with pandas](#6-basic-analysis-with-pandas)
- [7 Visualization of structured data](#7-visualization-of-structured-data)
- [8 Histograms: Identifying distributions](#8-histograms-identifying-distributions)
- [9 Boxplots: Identifying outliers](#9-boxplots-identifying-outliers)
- [10 Scatterplots: Recognizing connections](#10-scatterplots-recognizing-connections)
- [11 Heatmaps: Visualizing correlations](#11-heatmaps-visualizing-correlations)
- [12 Visualization of semi-structured and unstructured data](#12-visualization-of-semi-structured-and-unstructured-data)
    - [Text data: Wordclouds](#text-data-wordclouds)
    - [Image data: Pixel values as a matrix](#image-data-pixel-values-as-a-matrix)
- [13 Missing values: Why do they occur?](#13-missing-values-why-do-they-occur)
- [14 Types of missing values: MCAR, MAR, MNAR](#14-types-of-missing-values-mcar-mar-mnar)
- [15 Consequences of missing values](#15-consequences-of-missing-values)
- [16 Analyze missing values](#16-analyze-missing-values)
- [17 Methods for handling missing values](#17-methods-for-handling-missing-values)
- [18 Simple imputation with pandas](#18-simple-imputation-with-pandas)
- [19 Imputation with scikit-learn: SimpleImputer](#19-imputation-with-scikit-learn-simpleimputer)
- [20 Advanced imputation: KNNImputer](#20-advanced-imputation-knnimputer)
- [21 Missing indicator feature](#21-missing-indicator-feature)
- [22 Scaling and Normalization: Why?](#22-scaling-and-normalization-why)
    - [Scaling vs. Normalization](#scaling-vs-normalization)
- [23 Standardization: Z-Score transformation](#23-standardization-z-score-transformation)
- [24 Min-Max scaling](#24-min-max-scaling)
- [25 Logarithmic scaling](#25-logarithmic-scaling)
- [26 Which scaling method should be used and when?](#26-which-scaling-method-should-be-used-and-when)
- [27 Avoiding data leaks when scaling](#27-avoiding-data-leaks-when-scaling)
- [28 Preprocessing using a pipeline](#28-preprocessing-using-a-pipeline)
- [29 Sample workflow for the U phase](#29-sample-workflow-for-the-u-phase)
- [30 Common mistakes and best practices](#30-common-mistakes-and-best-practices)
    - [Error 1: Ignoring missing values](#error-1-ignoring-missing-values)
    - [Error 2: Data leakage due to incorrect preprocessing](#error-2-data-leakage-due-to-incorrect-preprocessing)
    - [Error 3: Choosing the wrong visualization](#error-3-choosing-the-wrong-visualization)
- [31 Best practice: Documentation](#31-best-practice-documentation)
- [32 Connection to MLOps](#32-connection-to-mlops)
- [33 Key terms](#33-key-terms)
- [34 Memorable quotes](#34-memorable-quotes)

---

## 1 Why is "Understanding the Data" so important?

The data phase is the foundation of a machine learning project.

**Before an algorithm is trained, the following must be clear:**
- what data is actually available
- what sources it comes from
- what structure it has
- what its quality is
- which values are missing or stand out
- which variables are correlated
- what preprocessing is necessary

**Important:** A model is only as good as the data it is trained on. Poor data does not lead to "creative" models, but rather to poor models with a false sense of mathematical confidence.

In practice, a large portion of the work in data science projects involves understanding and preparing data. Depending on the definition and study, approximately _45% to 80%_ of the workload is spent on data preparation, cleaning, exploration and preprocessing.

**This means:**
```text
Understand the data → Clean the data → Prepare the data → Model it effectively
```

Without this phase, subsequent steps such as algorithm selection, feature engineering, training and evaluation can hardly be carried out reliably.

---

## 2 Classification within the QUA³CK model

This file is part of the second phase of the QUA³CK process model.

| **Phase** | **Meaning**                                                       | **Role in this file**                                  |
| --------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| **Q**     | Question                                                          | Clarify the research question and objective            |
| **U**     | Understanding the Data                                            | Analyze, visualize and prepare data                    |
| **A³**    | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | Select models, tune features, optimize hyperparameters |
| **C**     | Conclude & Compare                                                | Evaluate and Compare Models                            |
| **K**     | Knowledge Transfer                                                | Documenting results and making them useful             |

**Note:** This file focuses primarily on the _U phase_. It serves as a bridge between the problem definition from [docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md](docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md) and the subsequent modeling.

---

## 3 Data sources in machine learning

Data can come from a wide variety of sources. For a machine learning project, it is important to document the origin and characteristics of the data.

| **Data Source**       | **Description**                      | **Examples**                                    |
| --------------------- | ------------------------------------ | ----------------------------------------------- |
| Relational Databases  | Structured Data in Table Form        | PostgreSQL, MySQL, Oracle                       |
| APIs & Web Scraping   | Data from interfaces or websites     | REST APIs, Twitter/X API, BeautifulSoup         |
| Public Datasets       | Freely Available Data Sets           | UCI Repository, Kaggle, Google Dataset Search   |
| Internal company data | Data from operational systems        | CRM, ERP, server logs, proprietary databases    |
| Unstructured data     | Data without a fixed table structure | Text, images, videos, audio, social media posts |

**Tip:** For your portfolio or course project, you should always clearly document the data source: origin, URL, license, collection method and time period.

---

## 4 Understanding data types

Not all data is organized in the same structure.

### Structured data

Structured data has a fixed tabular format.

**Examples:**
- CSV files
- Excel spreadsheets
- SQL databases
- Tables with clearly defined columns and data types

**Typical processing:**
```python
import pandas as pd

df = pd.read_csv("dataset.csv")
df.info()
df.describe()
```

### Semi-structured data

Semi-structured data has a certain degree of order but does not follow a traditional tabular structure.

**Examples:**
- JSON
- XML
- NoSQL documents
- nested API responses

**Example:**
```json
{
    "kunde": {
        "id": 42,
        "alter": 31,
        "vertrag": "Premium"
    }
}
```

### Unstructured data

Unstructured data does not have a fixed, predetermined structure.

**Examples:**
- Texts
- PDFs
- Images
- Videos
- Audio files

**Important:** Unstructured data usually needs to be transformed first before traditional ML algorithms can process it. For example, text is vectorized and images are converted into pixel values or embeddings.

---

## 5 The goal of Data Exploration

Data exploration is often referred to as _Exploratory Data Analysis (EDA)_.

**It has four main objectives:**

| **Goal**                  | **Explanation**                                                  |
| ------------------------- | ---------------------------------------------------------------- |
| Check Data Quality        | Identify errors, inconsistencies, duplicates and invalid values  |
| Analyze Distributions     | Understanding the statistical properties of individual variables |
| Identify Outliers         | Identify and evaluate extreme or unusual values                  |
| Recognizing Relationships | Examining correlations and relationships between variables       |

**Note:** EDA is not just an optional "beautification" filter for charts. EDA is the step where you realize that the data isn't as clean as some optimistic CSV filename might have suggested.

---

## 6 Basic Analysis with pandas

The first look at the dataset is usually done using pandas.

```python
import pandas as pd

# Loading data
df = pd.read_csv("dataset.csv")

# Show Structure
print(df.info())

# Descriptive Statistics
print(df.describe())

# Missing values by column
print(df.isnull().sum())

# Percentage of missing values
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct)
```

**Key functions:**

| **Functions**    | **Purpose**                                           |
| ---------------- | ----------------------------------------------------- |
| `info()`         | shows data types and the number of non-missing values |
| `describe()`     | provides descriptive statistics                       |
| `isnull()`       | marks missing values                                  |
| `sum()`          | counts missing values per column                      |
| `value_counts()` | counts the frequencies of categorical values          |
| `corr()`         | calculates correlations between numerical features    |

---

## 7 Visualization of structured data

Visualizations help identify patterns that are easily overlooked in tables alone.

| **Visualization** | **Purpose**                                              | **Suitable for**         |
| ----------------- | -------------------------------------------------------- | ------------------------ |
| Histogram         | Plotting the distribution and frequency of a trait       | numerical features       |
| Boxplot           | Identifying outliers, medians, quartiles and variability | numerical features       |
| Scatterplot       | Examining the relationship between two variables         | two numerical features   |
| Heatmap           | Plotting correlations between multiple variables         | numerical feature matrix |

---

## 8 Histograms: Identifying distributions

A histogram shows how frequently certain ranges of values occur.

**Example:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x="Alter", bins=30, kde=True)
plt.title("Altersverteilung")
plt.show()
```

**What should you look for?**
- Is the distribution approximately normally distributed?
- Is it left- or right-skewed?
- Are there multiple peaks?
- Are there any noticeable gaps or clusters?

**Tip:** The KDE curve smooths out the distribution and makes trends easier to see.

---

## 9 Boxplots: Identifying outliers

A boxplot provides a concise summary of a numerical distribution.

**It shows:**
- the median
- the first and third quartiles
- the interquartile range
- typical dispersion
- potential outliers

```python
sns.boxplot(data=df, y="Einkommen")
plt.title("Einkommensverteilung und potenzielle Ausreißer")
plt.show()
```

**Interpretation:**

| **Element**     | **Meaning**                |
| --------------- | -------------------------- |
| Box             | the middle 50% of the data |
| Line in the box | Median                     |
| Whiskers        | typical spread             |
| Points outside  | potential outliers         |

**Important:** An outlier isn't automatically an error. A very high income could be a data error. But it could also simply be a very wealthy individual, because, unfortunately, that's how reality has been programmed.

---

## 10 Scatterplots: Recognizing connections

Scatterplots show the relationship between two numerical variables.

```python
sns.scatterplot(data=df, x="Alter", y="Einkommen")
plt.title("Zusammenhang zwischen Alter und Einkommen")
plt.show()
```

**Possible observations:**
- positive correlation
- negative correlation
- no discernible relationship
- clustering
- nonlinear relationships
- outliers

**Example interpretation:**
```text
If income tends to increase with age, there is a positive correlation.
```

---

## 11 Heatmaps: Visualizing correlations

A correlation heat map shows relationships between multiple numerical features.

```python
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
plt.title("Korrelationsmatrix aller numerischen Merkmale")
plt.show()
```

**Interpretation:**

| **Correlation coefficient** | **Meaning**                  |
| --------------------------- | ---------------------------- |
| close to **+1**             | strong positive relationship |
| close to **-1**             | strong negative relationship |
| close to **0**              | a barely linear relationship |

**Note:** Correlation does not imply causation. Just because two things occur together doesn't mean that one causes the other. Otherwise, ice cream sales would also be to blame for sunburn. Interpreting statistics remains a challenge.

---

## 12 Visualization of semi-structured and unstructured data

It's not just tabular data that can be visualized.

### Text data: Wordclouds

Word clouds show common terms in text data.

**Examples of use:**
- Customer reviews
- Social media comments
- Support tickets
- Free-text responses in surveys

More common words appear larger.

**Warning:** Wordclouds are good for a quick overview, but not very useful for precise analysis. For serious text analysis, additional methods such as tokenization, TF-IDF, or embeddings are more useful.

### Image data: Pixel values as a matrix

Images can be interpreted as numerical matrices. In grayscale images, each pixel corresponds to an intensity value.
```text
Image → pixel matrix → numerical data → model input
```

Heatmaps can help reveal patterns in pixel values.

---

## 13 Missing values: Why do they occur?

Missing values are very common in real-world datasets.

**Typical causes:**

| **Cause**               | **Explanation**                                               |
| ----------------------- | ------------------------------------------------------------- |
| Survey error            | technical errors, incorrect data entry, measurement problems  |
| System-related failures | Sensor errors, server outages, transmission problems          |
| Intentional omissions   | Privacy, irrelevant questions, refusal to provide information |

**Important:** Missing values must not simply be ignored. They can skew analyses, compromise models and lead to incorrect decisions.

---

## 14 Types of missing values: MCAR, MAR, MNAR

To handle missing values effectively, you need to understand why they are missing.

| **Type** | **Meaning**                  | **Example**                                                                 |
| -------- | ---------------------------- | --------------------------------------------------------------------------- |
| **MCAR** | Missing Completely At Random | Values are missing purely by chance, e.g. due to a random technical failure |
| **MAR**  | Missing At Random            | Absenteeism depends on other observable variables                           |
| **MNAR** | Missing Not At Random        | The absence is related to the missing value itself                          |

**Examples:**
- _MCAR:_ Some observed values are missing by chance due to a brief system error.
- _MAR:_ Younger customers are less likely to report their income, but their age is known.
- _MNAR:_ People with very high incomes deliberately do not report their income.

**Tip:** The type of missing data determines which imputation method is appropriate.

---

## 15 Consequences of missing values

Missing values can cause several problems.

| **Consequence**   | **Explanation**                                                           |
| ----------------- | ------------------------------------------------------------------------- |
| Biased Analyses   | Statistical measures such as the mean or standard deviation are distorted |
| Model error       | Many ML algorithms cannot handle `NaN` directly                           |
| Worse Performance | Information loss reduces the quality of predictions                       |
| Bias              | Systematically missing values can put certain groups at a disadvantage    |

**Warning:** If missing values occur systematically, the problem is not just technical, but analytical. In that case, a quick `fillna()` isn't enough, even if it feels good in terms of productivity.

---

## 16 Analyze missing values

Before imputing, you should identify the missing values.

```python
# Number of missing values per column
missing_count = df.isnull().sum()

# Percentage
missing_pct = (missing_count / len(df)) * 100
missing_summary = pd.DataFrame({
"missing_count": missing_count,
"missing_pct": missing_pct
})

print(missing_summary.sort_values("missing_pct", ascending=False))
```

**Possible guiding questions:**
- Which features have missing values?
- What is the percentage of missing values for each feature?
- Are the missing values random or systematic?
- Are there any groups with a particularly high number of missing values?
- Should a feature be removed, imputed, or flagged?

---

## 17 Methods for handling missing values

There are various strategies.

| **Method**        | **Idea**                                | **Suitable for**                 | **Risk**                         |
| ----------------- | --------------------------------------- | -------------------------------- | -------------------------------- |
| Delete rows       | Removes data points with missing values | very low error rate              | Loss of information              |
| Delete columns    | Removes entire features                 | a large number of missing values | Important feature may be lost    |
| Mean imputation   | replaces missing values with the mean   | numeric data, MCAR               | skewed distribution              |
| Median imputation | replaced by the median                  | numerical data with outliers     | reduces variance                 |
| Mode imputation   | replaced with the most frequent value   | categorical data                 | may reinforce the dominant class |
| KNN imputation    | uses similar data points                | correlated features              | more computationally intensive   |
| MICE              | multiple iterative imputation           | MAR Data                         | more complex                     |
| MissForest        | Random Forest-based imputation          | mixed data types                 | high computational effort        |

---

## 18 Simple imputation with pandas

**Example:** Mean-value imputation for a numeric column.
```python
# Counting missing values
print(df["Einkommen"].isnull().sum())

# Calculate the mean
mean_income = df["Einkommen"].mean()

# Replace Missing Values
df["Einkommen"] = df["Einkommen"].fillna(mean_income)

# Check the result
print(df["Einkommen"].isnull().sum())
```

**Warning:** Mean imputation is simple, but it is not necessarily a good method. It can smooth distributions, reduce variance, and weaken correlations.

---

## 19 Imputation with scikit-learn: SimpleImputer

```python
from sklearn.impute import SimpleImputer

# Mean Imputation
imputer_mean = SimpleImputer(strategy="mean")
df[["Alter", "Einkommen"]] = imputer_mean.fit_transform(
df[["Alter", "Einkommen"]]
)

# Median-Imputation
imputer_median = SimpleImputer(strategy="median")

# Mode Imputation for Categorical Data
imputer_mode = SimpleImputer(strategy="most_frequent")
```

**Tip:** For numerical data containing outliers, the median is often more robust than the mean.

---

## 20 Advanced imputation: KNNImputer

KNN imputation replaces missing values using similar data points.

```python
from sklearn.impute import KNNImputer
import pandas as pd

knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = knn_imputer.fit_transform(df)
df = pd.DataFrame(df_imputed, columns=df.columns)
```

**Advantages:**
- Takes into account relationships between features
- Uses information from similar data points
- Often produces more realistic values than simple imputation

**Disadvantages:**
- Requires numeric or appropriately coded data
- More computationally intensive with large datasets
- Sensitive to unscaled data

**Important:** KNN imputation should generally be used after appropriate preparation of numerical features, because otherwise distance calculations may be dominated by large value ranges.

---

## 21 Missing indicator feature

Sometimes it's not just the value that matters, but also the fact that a value is missing.

**Example:**
```python
df["Einkommen_missing"] = df["Einkommen"].isnull().astype(int)
```

This can be useful when missing values occur systematically.

**Example:**
```text
Customers without a satisfaction score may have less contact with the company. The absence itself then conveys information.
```

---

## 22 Scaling and Normalization: Why?

Many ML algorithms are sensitive to features that have very different ranges of values.

**Example:**

| **Feature** | **Value range**   |
| ----------- | ----------------- |
| Age         | 18 to 90          |
| Income      | 20,000 to 500,000 |

Without scaling, income dominates distance calculations, although age could also be a relevant factor.

**The following are particularly affected:**
- k-nearest neighbors
- k-means
- support vector machines
- neural networks
- PCA
- linear and logistic regressions with regularization

**The following are often less sensitive:**
- Decision Trees
- Random Forests
- Gradient Boosting Models

**Note:** Tree-based methods partition data based on thresholds and are therefore often more robust to different scales. Distance-based methods, on the other hand, suffer immediately when one feature is significantly larger than the rest.

### Scaling vs. Normalization

The terms are often used interchangeably, but they do not mean exactly the same thing.

| **Term**        | **Meaning**                                                     |
| --------------- | --------------------------------------------------------------- |
| Scaling         | General term for transformations to comparable ranges of values |
| Standardization | Transformation to a mean of 0 and a standard deviation of 1     |
| Normalization   | frequently transformed to the range `[0, 1]` or `[-1, 1]`       |

**Important:** In practice, it is not so much the term itself that matters, but rather the specific method and its effect on the data.

---

## 23 Standardization: Z-Score transformation

Standardization transforms values so that the new distribution has a mean of 0 and a standard deviation of 1.

```text
z = (x - μ) / σ
```

| **Symbol** | **Meaning**        |
| ---------- | ------------------ |
| `x`        | Original value     |
| `μ`        | Mean               |
| `σ`        | Standard deviation |
| `z`        | Standardized value |

**Properties:**
- The mean becomes 0
- The standard deviation becomes 1
- The original distribution shape is preserved
- The range of values is not limited

**Suitable for:**
- linear regression
- logistic regression
- PCA
- SVM
- k-NN
- normally distributed or approximately normally distributed data

**Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 24 Min-Max scaling

Min-max scaling transforms values into a fixed range, usually `[0, 1]`.

```text
x_scaled = (x - x_min) / (x_max - x_min)
```

**Properties:**
- Values range from 0 to 1
- Relative distances remain proportional
- Easy to interpret
- Sensitive to outliers

**Suitable for:**
- Neural networks
- k-NN
- k-means
- Data with a known range of values

**Example:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Warning:** A single extreme outlier can significantly distort the min-max scaling.

---

## 25 Logarithmic scaling

Logarithmic scaling is particularly suitable for distributions that are heavily skewed to the right.

```text
x_log = log(x + 1)
```

Adding 1 prevents problems with `log(0)`.

**Typical applications:**
- Income
- Prices
- Website traffic
- Population figures
- Transaction amounts

**Example:**
```python
import numpy as np

df["Einkommen_log"] = np.log1p(df["Einkommen"])
```

**Advantages:**
- reduces the influence of outliers
- makes right-skewed distributions more symmetrical
- can improve model performance

---

## 26 Which scaling method should be used and when?

| **Situation**                    | **Appropriate method**                |
| -------------------------------- | ------------------------------------- |
| Normally distributed data        | Standardization                       |
| Distance-based algorithms        | Standardization or Min-Max scaling    |
| Neural networks                  | Min-Max scaling or standardization    |
| Data heavily skewed to the right | Log scaling or Box-Cox transformation |
| Many outliers                    | Robust scaling or log transformation  |
| Tree-based models                | Scaling is often not necessary        |

**Tip:** Scaling is not an end in itself. What matters is which algorithm is used and what the data distribution is.

---

## 27 Avoiding data leaks when scaling

A very common mistake is to refit scalers to test data.

**Wrong:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# WRONG: fit_transform on test data
X_test_scaled = scaler.fit_transform(X_test)
```

**Correct:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit Based Only on Training Data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data only
X_test_scaled = scaler.transform(X_test)
```

**Why?**
- The scaler's parameters - such as the mean and standard deviation - must be learned solely from the training data.

**Danger:** When test data is used during fitting, information from the test data is incorporated into the training process. The evaluation then appears better than it actually is. The model isn't cheating. It's just that the human ruined the experimental setup.

---

## 28 Preprocessing using a pipeline

In practice, preprocessing should be organized as part of a pipeline whenever possible.

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

**Advantages:**
- Less data leakage
- More reproducible
- Cleaner structure
- Easier to combine with cross-validation
- Better suited for later production

---

## 29 Sample workflow for the U phase

**A useful workflow for "Understanding the Data" might look like this:**
```text
1. Load the dataset
2. Document the data source and license
3. Check the data structure
4. Analyze data types
5. Determine the target variable
6. Examine missing values
7. Check for duplicates and inconsistencies
8. Calculate descriptive statistics
9. Create visualizations
10. Evaluate outliers
11. Examine correlations
12. Define an imputation strategy
13. Select a scaling method
14. Document preprocessing
15. Save the prepared dataset for modeling
```

---

## 30 Common mistakes and best practices

### Error 1: Ignoring missing values

**Problem:**
- Models often cannot handle missing values
- Analyses are skewed
- Systematic patterns of missing values go undetected

**Better:**
- Always analyze missing values
- Check for patterns of missing values
- Justify the appropriate imputation strategy

### Error 2: Data leakage due to incorrect preprocessing

**Problem:**
- Scalers or imputation methods are fitted to test data
- Test information is indirectly incorporated into the model
- The evaluation results are unrealistically good

**Better:**
```python
# Correct principle
preprocessing.fit(X_train)
X_train_processed = preprocessing.transform(X_train)
X_test_processed = preprocessing.transform(X_test)
```

### Error 3: Choosing the wrong visualization

| **Wrong**                      | **Better**            |
| ------------------------------ | --------------------- |
| Histogram for categorical data | Bar chart             |
| Pie chart with many categories | Bar chart             |
| 3D chart with no added value   | Clear 2D chart        |
| Too many colors                | Limited color palette |

**Tip:** The visualization should be appropriate for the research question, not for the need to appear as scientific as possible.

---

## 31 Best practice: Documentation

Good data analysis is reproducible.

**Therefore, document the following:**
- Data source
- Dataset version
- Number of rows and columns
- Meaning of the features
- Target variable
- Missing values
- Identified outliers
- Imputation method used
- Scaling method used
- Rationale for each preprocessing decision
- Potential risks and assumptions

**Suitable tools:**
- Jupyter Notebook
- Git / GitHub
- README.md
- requirements.txt
- MLFlow
- Data Sheets
- Model Cards

---

## 32 Connection to MLOps

The U phase should be documented in such a way that future experiments can be reproduced.

| **Task**               | **Relation to MLOps**             |
| ---------------------- | --------------------------------- |
| Save data version      | Reproducibility                   |
| Document preprocessing | Comparability                     |
| Use pipelines          | Fewer manual errors               |
| Save parameters        | Traceability                      |
| Versioning Notebooks   | Ability to work as part of a team |

**Important:** A preprocessing step that isn't documented is practically nonexistent. Or worse: It exists somewhere in a notebook cell that no one will ever execute in the correct order again.

---

## 33 Key terms

| **Term**               | **Brief Definition**                                                  |
| ---------------------- | --------------------------------------------------------------------- |
| **EDA**                | Exploratory data analysis to examine structure, quality, and patterns |
| **Feature**            | Input variable of a model                                             |
| **Target / Label**     | Target variable to be predicted                                       |
| **Imputation**         | Replacing missing values with estimated or calculated values          |
| **KNN-Imputation**     | Imputation based on similar data points                               |
| **MICE**               | Multiple Imputation by chained Equations                              |
| **MissForest**         | Random Forest-based imputation                                        |
| **Scaling**            | Transforming features into comparable value ranges                    |
| **Standardization**    | Transformation to a mean of 0 and a standard deviation of 1           |
| **Normalization**      | Transformation to a fixed range, often `[0, 1]`                       |
| **Data Leakage**       | Information flow from test data during training or preprocessing      |
| **Correlation Matrix** | Matrix of pairwise correlations between numerical variables           |
| **Outlier**            | A data point that deviates significantly from the majority of values  |
| **Pipeline**           | Chaining of preprocessing and modeling steps                          |

---

## 34 Memorable quotes

1. _Quote:_ The U phase prevents blind modeling.
2. _Quote:_ EDA reveals the problems and patterns hidden in the data.
3. _Quote:_ Missing values are not just gaps; they are often indicators of the data collection process.
4. _Quote:_ Outliers must be evaluated before they are removed.
5. _Quote:_ Scaling is particularly important for distance-based algorithms and neural networks.
6. _Quote:_ `fit()` is used on training data and `transform()` is used on both training and test data.
7. _Quote:_ Documentation is not a bonus; it is part of the analysis.
