# QUA³CK prozess model for Data-Science-Projects

> **Summary:**
>
> This file introduces the _QUA³CK process model_ as a structured framework for machine learning and data science projects. The focus is not only on the technical implementation of an ML project, but also on systematically planning it from the _research question_ through _data analysis_ and _model development_ to _production deployment_.

**ToC:**
- [1 Why are structured ML processes necessary?](#1-why-are-structured-ml-processes-necessary)
- [2 An Overview of the QUA³CK Model](#2-an-overview-of-the-qua³ck-model)
- [3 Q - Question: Understand the problem](#3-q---question-understand-the-problem)
    - [Q phase: Example - Iris Project](#q-phase-example---iris-project)
- [4 U - Understanding the data: Understanding data](#4-u---understanding-the-data-understanding-data)
    - [U phase: Example - Iris Project](#u-phase-example---iris-project)
- [5 A³ - Developing and optimizing Algorithms](#5-a³---developing-and-optimizing-algorithms)
    - [A³ phase: Example - Iris Project](#a³-phase-example---iris-project)
- [6 Key difference: X/y Split vs. Train/Test Split](#6-key-difference-xy-split-vs-traintest-split)
    - [X/y Split](#xy-split)
    - [Train/Test Split](#traintest-split)
- [7 MLOps: Making experiments reproducible](#7-mlops-making-experiments-reproducible)
- [8 C - Conclude & Compare: Evaluate models](#8-c---conclude--compare-evaluate-models)
    - [Quantitative Criteria](#quantitative-criteria)
    - [Qualitative Criteria](#qualitative-criteria)
    - [C phase: Example - Iris Project](#c-phase-example---iris-project)
- [9 K - Knowledge Transfer: Putting results to use](#9-k---knowledge-transfer-putting-results-to-use)
- [10 From Analysis to Production](#10-from-analysis-to-production)
- [11 Portfolio Relevance](#11-portfolio-relevance)
- [12 Key Terms](#12-key-terms)
- [13 Memorable quotes](#13-memorable-quotes)

---

## 1 Why are structured ML processes necessary?

Many data science projects fail not because of a lack of algorithms, but because of a lack of structure.

**Common problems include:**
- Unclear problem definition
- Lack of success metrics
- Lack of proper documentation
- Experiments cannot be reproduced
- Models remain in the notebook and never make it into production

**Important:** A good ML project doesn't start with code, but with a clear research question.

The QUA³CK model helps systematically implement data science projects from concept to application.

---

## 2 An Overview of the QUA³CK Model

The _QUA³CK model_ was developed as a practice-oriented process model for machine learning projects. It combines a scientific approach with practical applicability.

**QUA³CK consists of five phases:**

| **Phase** | **Meaning**                                                       | **Key Question**                                       |
| --------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| **Q**     | Question                                                          | What problem needs to be solved?                       |
| **U**     | Understanding the Data                                            | What is the structure and quality of the data?         |
| **A³**    | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | Which model works best?                                |
| **C**     | Conclude & Compare                                                | Which model is the best overall?                       |
| **K**     | Knowledge Transfer                                                | How are results documented and made available for use? |

**Note:** The process is not strictly linear. The _A³ phase_, in particular, is iterative: models are trained, adjusted, evaluated and improved.

---

## 3 Q - Question: Understand the problem

The first phase is the most important foundation of the entire project.

**The Q phase defines:**
- the specific problem to be solved
- the target audience
- the applicable success metrics
- the desired outcome or product

### Q phase: Example - Iris Project

| **Aspect**             | **Example**                                |
| ---------------------- | ------------------------------------------ |
| **Problem**            | Automatic Classification of Iris Species   |
| **Target Audience**    | Botany students or field researchers       |
| **Data Base**          | Flower Characteristics of the Iris Dataset |
| **Performance metric** | Accuracy > 95 %                            |
| **Deployment Target**  | Interactive Streamlit App                  |

**Important:** Without a clear research question, a technically sound model can still be of no practical use.

---

## 4 U - Understanding the data: Understanding data

During the U phase, the data is analyzed before a model is trained.

**Objectives of this phase:**
- Understand the data structure
- Identify missing or outlier values
- Analyze distributions
- Identify relationships between features
- Derive initial hypotheses for modeling

**Typical methods:**
- descriptive statistics
- scatter plots
- box plots
- correlation analysis
- testing of class distributions

### U phase: Example - Iris Project

**The Iris dataset contains:**

| **Component**   | **Description**                                      |
| --------------- | ---------------------------------------------------- |
| Observations    | 150 iris flowers                                     |
| Classes         | Setosa, Versicolor, Virginica                        |
| Features        | sepal length, sepal width, petal length, petal width |
| Target variable | Iris Species                                         |

**Key findings from the EDA:**
- _Petal length_ and _petal width_ distinguish between iris species much more effectively than sepal characteristics.
- Setosa is easily distinguishable.
- Versicolor and Virginica overlap more significantly.

**Tip:** EDA often determines which features are important and which models might be useful.

---

## 5 A³ - Developing and optimizing Algorithms

The A³ phase is the actual modeling phase.

**A³ stands for:**

| **Component**             | **Meaning**                     |
| ------------------------- | ------------------------------- |
| Algorithm Selection       | select appropriate algorithms   |
| Adapting Features         | Customize or transform features |
| Adjusting Hyperparameters | Optimize Hyperparameters        |

This phase is repeated several times. The goal is to systematically test and improve various models.

### A³ phase: Example - Iris Project

**The "Big 3" in the Iris Project (a comparative analysis of three approaches):**

| **Model**           | **Typ**                | **Idea**                                  |
| ------------------- | ---------------------- | ----------------------------------------- |
| Decision Tree       | supervised learning    | Decision Rules as a Tree                  |
| K-Nearest Neighbors | supervised learning    | Classification Based on Similar Neighbors |
| K-Means             | self-directed learning | Grouping of Similar Data Points           |

---

## 6 Key difference: X/y Split vs. Train/Test Split

**X/y Split vs. Train/Test Split:** These two terms are often confused.

### X/y Split

**In an _X/y Split_, features and the target variable are separated:**

| **Symbol** | **Meaning**                |
| ---------- | -------------------------- |
| **X**      | Input Variables / Features |
| **y**      | Target variable / Label    |

**Example:**
```python
X = df.drop("species", axis=1)
y = df["species"]
```

### Train/Test Split

**In a _train/test split_, the data points are divided into training and test data:**

| **Data Section**  | **Purpose**                         |
| ----------------- | ----------------------------------- |
| **Training Data** | The model learns from this          |
| **Test data**     | Independent evaluation of the model |

**Example:**
```python
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
```

**Note:** _X/y Split_ splits columns by role. _Train/Test Split_ splits rows by usage.

---

## 7 MLOps: Making experiments reproducible

MLOps stands for _Machine Learning Operations_.

**Goals of MLOps:**
- Document experiments in a traceable manner
- Save parameters and metrics
- Systematically compare models
- Prepare for deployment
- Facilitate collaboration

In this course, _MLFlow_ is particularly relevant for this purpose.

**With MLFlow, you can log:**
- parameters used
- metrics achieved
- model versions
- artifacts
- experiment runs

**Important:** MLOps ensures that an ML project not only works once, but remains reproducible, comparable, and scalable.

---

## 8 C - Conclude & Compare: Evaluate models

In Phase C, the results of the experiments are compared. It's not just the highest accuracy that matters. A model must also be practical.

### Quantitative Criteria

| **Metric**     | **Meaning**                                 |
| -------------- | ------------------------------------------- |
| Accuracy       | Percentage of Correct Predictions           |
| Precision      | How many positive predictions were correct? |
| Recall         | How many actual positive cases were found?  |
| F1-Score       | Balancing Precision and Recall              |
| Inference time | Speed of a Forecast                         |

### Qualitative Criteria

| **Criterion**         | **Meaning**                          |
| --------------------- | ------------------------------------ |
| Interpretability      | Can we understand model decisions?   |
| Complexity            | How complex is the model?            |
| Maintainability       | How easy will it be to adjust later? |
| Deployment Capability | Can it be used effectively?          |

### C phase: Example - Iris Project

**Sample Results from the Iris Project:**

| **Model**           | **Metric**          | **Result** |
| ------------------- | ------------------- | ---------- |
| Decision Tree       | Accuracy            | 97.8 %     |
| K-Nearest Neighbors | Accuracy            | 97.8 %     |
| K-Means             | Adjusted Rand Score | 0.669      |

Although decision trees and KNN perform equally well, the _decision tree_ may be preferred because it can be more interpretable and efficient.

**Tip:** The best model is not necessarily the most complex one, but rather the one that offers the best balance of performance, clarity and practicality.

---

## 9 K - Knowledge Transfer: Putting results to use

The K phase focuses on presenting the project's results in a clear and usable format.

**Possible deliverables:**
- Documented Jupyter Notebook
- GitHub repository
- Streamlit web app
- Project report
- Portfolio entry
- Presentation for stakeholders

In an academic context, the focus is on clear documentation. In practice, it is also important that a model can be used productively.

**Important:** An ML project is only truly valuable when the results are communicated clearly and made practical.

---

## 10 From Analysis to Production

**This modern approach combines QUA³CK with MLOps:**

| **QUA³CK-Phase** | **Traditional Approach** | **Modern MLOps Approach**  |
| ---------------- | ------------------------ | -------------------------- |
| Q + U            | Static laptops           | Interactive analysis apps  |
| A³               | local experiments        | MLFlow Experiment Tracking |
| C                | manual reports           | automated model comparison |
| K                | on-premises deployment   | GitHub + Streamlit Cloud   |

**This creates a connection between:**
```text
Problem statement → Data understanding → Modeling → Evaluation → Deployment
```

---

## 11 Portfolio Relevance

For a portfolio, a project should not only include code but also illustrate the entire process.

**A well-structured portfolio demonstrates:**

| **Section** | **Contents**                                   |
| ----------- | ---------------------------------------------- |
| Problem     | What needs to be solved?                       |
| Data        | What data was used?                            |
| EDA         | What patterns were identified?                 |
| Models      | Which algorithms were tested?                  |
| Evaluation  | Which model was the best, and why?             |
| Deployment  | How can the solution be used?                  |
| Reflection  | What have we learned? What are the next steps? |

**Example of the Iris Project:**

| **Aspect**     | **Contents**                                |
| -------------- | ------------------------------------------- |
| Project        | AMALEA QUA³CK Demo - Iris Classification    |
| Methodology    | QUA³CK Process Model                        |
| Best Algorithm | Decision Tree                               |
| Performance    | 97.8 % Accuracy                             |
| Technologies   | Python, Pandas, Scikit-learn, Matplotlib    |
| Next Steps     | MLFlow Integration and Streamlit Deployment |

---

## 12 Key Terms

| **Term**                | **Brief Definition**                                                              |
| ----------------------- | --------------------------------------------------------------------------------- |
| **EDA**                 | Exploratory data analysis to examine the structure, quality, and patterns in data |
| **Feature**             | Input variable used by the model for predictions                                  |
| **Label**               | Target variable to be predicted                                                   |
| **Hyperparameter**      | Settings to be configured before the workout                                      |
| **Overfitting**         | The model learns the training data too closely and generalizes poorly             |
| **Underfitting**        | The model is too simple and does not recognize patterns well enough               |
| **Train/Test Split**    | Split into training data and independent test data                                |
| **Experiment Tracking** | Systematic Documentation of Model Runs                                            |
| **Deployment**          | Providing a model for practical use                                               |
| **Reproducibility**     | Ability to reproduce results using the same data and the same code                |

---

## 13 Memorable quotes

1. _Quote:_ QUA³CK is a structured process that takes you from the initial question to the application.
2. _Quote:_ The Q phase determines whether the right problem is being solved.
3. _Quote:_ The U phase prevents blind modeling.
4. _Quote:_ The A³ phase is iterative: select, adapt, optimize.
5. _Quote:_ The C phase is based not only on accuracy, but also on overall quality.
6. _Quote:_ The K phase turns an experiment into a usable result.
