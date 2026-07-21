# The machine learning environment

> **Summary:**
>
> This file lays the conceptual foundation upon which the A³ phase of the QUA³CK model is built: what machine learning _actually_ is, what basic types of systems exist and why models fail—whether due to poor data or poor algorithms. Anyone who does not have a solid grasp of the concepts covered in this file will end up guessing rather than making informed decisions, at the latest when selecting algorithms in A³.

**ToC:**
- [1 Why is it important to understand the ML environment?](#1-why-is-it-important-to-understand-the-ml-environment)
- [2 Classification within the QUA³CK model](#2-classification-within-the-qua³ck-model)
- [3 What is machine learning?](#3-what-is-machine-learning)
- [4 Why machine learning?](#4-why-machine-learning)
- [5 An Overview of types of ML systems](#5-an-overview-of-types-of-ml-systems)
- [6 Supervised learning](#6-supervised-learning)
- [7 Self-directed learning](#7-self-directed-learning)
- [8 Self-Supervised learning and Reinforcement learning](#8-self-supervised-learning-and-reinforcement-learning)
- [9 Batch learning vs. Online learning](#9-batch-learning-vs-online-learning)
- [10 Instance-based vs. Model-based learning](#10-instance-based-vs-model-based-learning)
- [11 The biggest challenges: poor data](#11-the-biggest-challenges-poor-data)
- [12 The biggest challenges: poor algorithms](#12-the-biggest-challenges-poor-algorithms)
- [13 Testing and Validation](#13-testing-and-validation)
- [14 The No-Free-Lunch Theorem](#14-the-no-free-lunch-theorem)
- [15 Key terms](#15-key-terms)
- [16 Memorable quotes](#16-memorable-quotes)

---

## 1 Why is it important to understand the ML environment?

Before selecting an algorithm, you should understand the conceptual framework you're working within. The field of machine learning is vast: supervised and unsupervised learning, batch and online learning, instance-based and model-based methods - and on top of that, there are terms like overfitting, underfitting, validation dataset and hyperparameters, which are assumed to be familiar in virtually every subsequent file.

**Note:** This file intentionally contains very little code. The focus is on the map, not the toolbox - the tools will be covered in the following files.

---

## 2 Classification within the QUA³CK model

This file provides the basic concepts on which two QUA³CK phases are based.

| **Phase** | **Meaning**                                                       | **Role in this file**                                                                                                     |
| --------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Q**     | Question                                                          | Not directly affected                                                                                                     |
| **U**     | Understanding the Data                                            | The understanding of data gained in the U phase helps determine which type of ML system is appropriate in the first place |
| **A³**    | Algorithm Selection, Adapting Features, Adjusting Hyperparameters | The learning methods presented here are the building blocks for future algorithm selection                                |
| **C**     | Conclude & Compare                                                | Not directly affected                                                                                                     |
| **K**     | Knowledge Transfer                                                | Not directly affected                                                                                                     |

**Note:** This file is fundamental to both _U_ ([docs/course-material/003_Understanding_the_data.md](docs/course-material/003_Understanding_the_data.md)) and _A³_: Without distinguishing between supervised, unsupervised, reinforcement, etc., it is impossible to meaningfully discuss "algorithm selection" in the A³ phase of [docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md](docs/course-material/002_QUACK_prozess_model_for_Data_Science_Projects.md).

---

## 3 What is machine learning?

Machine learning can easily be described as the science (and art) of programming computers to learn from data, rather than being explicitly programmed for each individual case.

**Two classic definitions continue to shape the discourse to this day:**
- Arthur Samuel (1959): Machine learning is the field that gives computers the ability to learn without being explicitly programmed.
- Tom Mitchell (1997): A program learns from experience E with respect to a task T and a performance metric P if its performance on T, as measured by P, improves as it gains experience E.

**Take spam filters as an example:** The task T is to detect new spam emails; the experience E consists of the training examples labeled as spam or ham; and the performance metric P could be the proportion of correctly classified emails (accuracy). The part of the system that learns and makes predictions is called the model.

**Important:** Simply storing data does not constitute machine learning. A computer that downloads a complete copy of Wikipedia will know more characters afterward, but it won't get any better at any specific task—it simply lacks the task T.

**The key difference from traditional programming:**
- Instead of writing rules by hand and tweaking them for every exception, the ML system derives patterns directly from the data.
- The program remains shorter, easier to maintain and adapts as the underlying data changes.

---

## 4 Why machine learning?

For many tasks, traditional, rule-based programming results in long lists of fragile rules that must be rewritten every time an exception occurs (a classic example: spam words like "For You" are blocked, so spammers switch to "4U" - and the rules start all over again). ML systems, on the other hand, learn such patterns directly from sample data and automatically adapt to new variations.

| **Scope**                                | **ML Task**                 | **Typical Technique**                                          |
| ---------------------------------------- | --------------------------- | -------------------------------------------------------------- |
| Automatically classify product images    | Image classification        | Convolutional Neural Networks (CNNs), Transformer              |
| Detecting tumors in brain scans          | Semantic image segmentation | CNNs, Transformer                                              |
| Classify news articles                   | Text classification (NLP)   | RNNs, CNNs, Transformer                                        |
| Predicting company revenue for next year | Regression                  | linear/polynomial regression, random forest, neural networks   |
| Recognizing credit card fraud            | Anomaly Detection           | Isolation Forest, Gaussian Mixture Distributions, Autoencoders |
| Segment customers by purchasing behavior | Clustering                  | k-Means, DBSCAN                                                |
| Visualizing high-dimensional data        | Size (dimension) Reduction  | PCA, t-SNE                                                     |
| Generate product recommendations         | Recommendation System       | Neural networks based on purchase histories                    |
| Build an intelligent game bot            | Reinforcement Learning      | Policy-Optimierung (z.B. AlphaGo)                              |

**Tip:** Machine learning really shines in situations where traditional solutions would either have to consist of many specialized rules, where no known algorithm exists for complex problems, or where the environment is constantly changing and a static set of rules would quickly become outdated.

---

## 5 An Overview of types of ML systems

ML systems can be classified according to several criteria that are not mutually exclusive.

| **Criterion**          | **Manifestations**                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------- |
| Training Monitoring    | supervised, unsupervised, self-supervised, semi-supervised, reinforcement learning |
| Incremental Learning   | Batch learning (offline) vs. Online learning (incremental)                         |
| Type of generalization | Instance-based (comparison with known examples) vs. model-based (prediction model) |

**These criteria can be combined in any way:** For example, a modern spam filter continuously learns using a neural network trained on examples flagged by users - meaning it is simultaneously model-based, supervised and online.

---

## 6 Supervised learning

In supervised learning, the training data already contains the desired solutions, known as labels.

**The two most common tasks are:**
- _Classification:_ Assigning data points to discrete categories, e.g. spam vs. ham.
- _Regression:_ Predicting a numerical target value, e.g. the price of a used car based on features such as mileage, age, and make (known as predictors).

**Example algorithms:** linear and logistic regression, k-nearest neighbors, support vector machines, decision trees and random forests, neural networks.

**Note:** "Target" and "Label" are often used interchangeably. However, the standard convention is: "Target" for regression tasks and "Label" for classification. Depending on the context, features are also referred to as predictors or attributes.

---

## 7 Self-directed learning

In unsupervised learning, the training data is unlabeled - the system must discover patterns on its own, without guidance.

| **Technique**              | **Goal**                                                            | **Example**                                            |
| -------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------ |
| Clustering                 | Group similar data points                                           | Identifying a blog's visitor groups                    |
| Size Reduction             | Simplify features without losing much information                   | Aggregate a car's mileage and age into "wear and tear" |
| Visualization              | Making high-dimensional data visualizable in 2D/3D                  | t-SNE Visualization of semantic clusters               |
| Anomaly Detection          | Identify unusual data points                                        | Credit card fraud, manufacturing defects               |
| Novelty Detection          | Detect novel instances that were completely unknown during training | New object class in pictures                           |
| Learning Association Rules | Find interesting relationships between characteristics              | Shopping cart analysis at the supermarket              |

**Tip:** Feature reduction is often worthwhile as a preprocessing step before applying a supervised learning algorithm: fewer features usually mean faster training and lower memory requirements and sometimes even better results.

---

## 8 Self-Supervised learning and Reinforcement learning

_Self-supervised learning_ automatically generates labels from what is essentially an unlabeled dataset - for example, by obscuring part of an image and training the model to reconstruct the original. The trained model is usually adapted to the actual target task using transfer learning and fine-tuned with a small number of labeled examples. Because it uses labels generated during training but is based on an unlabeled raw dataset, it is best considered a separate category between supervised and unsupervised learning.

_Reinforcement learning_ works fundamentally differently: An agent observes an environment, selects actions based on a policy, and receives rewards or penalties as a result. The goal is to find, over time, the policy that maximizes the cumulative rewards. AlphaGo, which defeated the world Go champion, is the best-known example.

**Note:** Semi-supervised learning also exists as a hybrid form: many unlabeled instances and a few labeled ones—for example, when photo services cluster faces (unsupervised) and then only one label is added per person.

---

## 9 Batch learning vs. Online learning

In _batch learning_, the system is trained offline using the entire available dataset and then runs unchanged in production - it does not continue to learn. Because the world continues to evolve but the model remains static, the quality deteriorates over time (model drift / data drift). The only solution is to retrain the model regularly using current data, which, however, requires time and computing power and reaches its limits when dealing with very large amounts of data.

In _online learning_, the system is trained incrementally, data point by data point or in small mini-batches. This is suitable for rapidly changing environments and for systems with limited resources. An important parameter is the learning rate: if set high, the system quickly forgets old patterns; if set low, it reacts more slowly but is more robust against noise and outliers.

**Important:** Algorithms for online learning are also suitable for _out-of-core learning_: Datasets that do not fit entirely into main memory are loaded and trained in chunks. Despite the name, this usually takes place offline - "online learning" here refers more to incremental learning than to an internet connection.

---

## 10 Instance-based vs. Model-based learning

In _instance-based learning_, the system essentially memorizes the training examples and generalizes to new cases using a similarity measure (e.g. k-nearest neighbors: a new instance belongs to the majority class of its most similar neighbors).

In _model-based learning_, on the other hand, a model with parameters is developed from the example data, and this model is then used for predictions.

**The typical workflow:**
1. Examine the data
2. Select a model (e.g. a linear model)
3. Define a cost function that measures how poorly the model fits the training data
4. Train the model - the algorithm searches for the parameters that minimize the cost function
5. Use the model to make predictions on new data (inference)

**A classic example:** A linear model "Satisfaction = θ₀ + θ₁ × GDP_per_capita" is trained on country-level data so that the two model parameters, θ₀ and θ₁, best fit the training data points. A prediction can then be derived from the GDP per capita for a country with no known satisfaction value.

**Note:** "Model" is an overloaded term: it can refer to a type of model (linear regression), a fully specified architecture, or the fully trained model with specific parameter values. Model selection means determining the type and architecture; training means finding the specific parameter values.

---

## 11 The biggest challenges: poor data

Since an ML system essentially consists of a model and training data, sources of error lie either in the model or in the data.

**When it comes to the data, four problems are particularly common:**

| **Problem**                                      | **Brief Description**                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insufficient amount of data                      | Most methods require thousands to millions of examples; for complex tasks such as image or speech recognition, even a simple algorithm can perform well with enough data ("the outrageous effectiveness of data")                                                                                          |
| Non-representative training data / Sampling bias | Training data must reflect the cases to which the model will later be generalized - otherwise, systematic biases will arise (a classic example: the 1936 Literary Digest poll, which incorrectly predicted Roosevelt's victory because the sample and response rate overrepresented wealthier respondents) |
| Poor data quality                                | Errors, outliers and noise make it difficult for the system to distinguish true patterns from randomness; a large part of data science work involves cleaning such data                                                                                                                                    |
| Irrelevant characteristics                       | Too many useless features or a lack of relevant ones will degrade the results ("garbage in, garbage out"); the solution is targeted feature engineering (selection, extraction, and creation of new features)                                                                                              |

**Important:** Even a very large sample can be biased if the survey method is flawed. Size alone does not protect against sampling bias - that was the problem as far back as 1936, not the sample size.

---

## 12 The biggest challenges: poor algorithms

In addition to poor data, the model itself can also be the problem - typically because its complexity is too high or too low relative to the data.

**Important:** _Overfitting:_ The model adapts too closely to the training data (including noise) and generalizes poorly to new data. This tends to occur with complex models or small or noisy training datasets.

| **Countermeasures for Overfitting** | **Idea**                                                               |
| ----------------------------------- | ---------------------------------------------------------------------- |
| Simplify the model                  | Fewer parameters, fewer features, tighter constraints (regularization) |
| Collect more training data          | Reduces the relative influence of random patterns                      |
| Reduce Noise                        | Correct data errors, remove outliers                                   |

The strength of regularization is controlled by a _hyperparameter_ - unlike a model parameter (which is learned during training), the hyperparameter is set before training and remains constant throughout training.

**Important:** _Underfitting:_ the exact opposite - the model is too simple to capture the structure of the data and produces inaccurate predictions even on the training data.

| **Countermeasures for Underfitting** | **Idea**                              |
| ------------------------------------ | ------------------------------------- |
| Choose a more powerful model         | More parameters, more capacity        |
| Provide better features              | Feature Engineering                   |
| Ease restrictions                    | Reduce regularization hyperparameters |

---

## 13 Testing and Validation

The only way to determine whether a model generalizes well is to test it on new, unseen data. Instead of risking this directly in the production system, the data is divided into a _training dataset_ and a _test dataset_ (often 80/20; for very large datasets, a smaller test portion is sufficient). The difference between the training error and the error on the test dataset (generalization error) indicates overfitting.

If multiple models or hyperparameter values are to be compared, a single test dataset is not sufficient - if it is repeatedly used for model selection, the model indirectly "learns" from it, leading to an overestimation of its actual quality. The solution is an additional _validation dataset_ (holdout validation): Multiple candidate models are trained on the training dataset, compared on the validation dataset and the best model is then retrained on the complete training dataset and evaluated once on the test dataset. If the validation dataset is too small, the model evaluation becomes inaccurate; if it is too large, there is not enough data for training. _Cross-validation_ using multiple small validation sets and averaging the results mitigates this dilemma, but requires proportionally more training time.

A special case is _data discrepancy_: training data (e.g. images of flowers from the web) systematically differ from the subsequent production data (e.g. photos from a mobile app). An additional _train-dev set_ can help here: a portion of the training data is set aside and tested after training. If the model performs poorly on the train-dev set, it is overfitting to the training data; if it performs well there but poorly on the actual validation dataset, the problem lies in the data discrepancy between the two data sources.

**Tip:** _Rule of thumb:_ Training dataset for learning, validation dataset (or train-dev set in case of data discrepancies) for comparing model candidates and test dataset for the single, final quality assessment.

---

## 14 The No-Free-Lunch Theorem

Every choice of model makes implicit assumptions about the data - a linear model, for example, assumes that the relationship between features is essentially linear and that deviations are merely noise. David Wolpert demonstrated in 1996 that, without any assumptions about the data, there is no reason to prefer one model over another (No-Free-Lunch Theorem). For some datasets, a linear model is optimal; for others, a neural network is—no model is a priori superior to all others.

**Note:** Since it is not possible in practice to test every conceivable model, one makes well-considered assumptions about the data and evaluates only a sensibly limited selection of models - which is exactly what happens in the A³ phase of QUA³CK, known as "Algorithm Selection".

---

## 15 Key terms

| **Term**                    | **Brief Definition**                                                                                  |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Feature**                 | Input variable used by the model for predictions                                                      |
| **Label**                   | Target variable in classification tasks                                                               |
| **Training dataset**        | Data used to train the model                                                                          |
| **Test dataset**            | Independent dataset for the final assessment of the generalization error                              |
| **Validation dataset**      | A subset of the training data set aside for comparing model candidates and hyperparameters            |
| **Train-Dev-Set**           | Additional dataset for distinguishing between overfitting and data discrepancies                      |
| **Hyperparameters**         | The learning algorithm setting, which is determined before training, remains constant during training |
| **Model parameters**        | A value that the learning algorithm determines on its own from the training data                      |
| **Overfitting**             | The model overfits the training data (including noise) and generalizes poorly                         |
| **Underfitting**            | The model is too simple to capture the structure of the data                                          |
| **Sampling Bias**           | Systematic bias caused by a flawed survey method, regardless of the sample size                       |
| **No-Free-Lunch-Theorem**   | Without making any assumptions about the data, no model is a priori superior to another               |
| **Batch learning**          | Training with the entire dataset at once, offline, no incremental learning                            |
| **Online learning**         | Incremental training using individual data points or mini-batches                                     |
| **Out-of-Core learning**    | Online learning technique for data sets that do not fit into main memory                              |
| **Instance-based learning** | Generalization by comparing new cases with stored examples using a similarity measure                 |
| **Model-based learning**    | Generalization using a predictive model trained on the data                                           |

---

## 16 Memorable quotes

1. _Quote:_ A model can only be as good as the task T, the experience E, and the performance metric P that are provided to it.
2. _Quote:_ "Garbage in, garbage out" applies equally to data volume, representativeness and feature selection.
3. _Quote:_ Overfitting learns the noise; underfitting overlooks the pattern.
4. _Quote:_ The test dataset should be used only once to reach a final conclusion - anyone who misuses it for model selection is only deceiving themselves.
5. _Quote:_ No model is always right: Without assumptions about the data, all models are equally good - and equally useless.
