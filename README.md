# Accenture Data Analytics - British Airways Customer Booking Behavior Analysis

This repository contains a Jupyter Notebook detailing a data science analysis focused on predicting customer booking behavior for British Airways. This project was completed as part of a job simulation exercise, demonstrating skills in data analysis, machine learning, and model evaluation.

## Project Overview

The goal of this analysis was to build a predictive model that determines whether a customer booking will be completed. This is a crucial task for airlines like British Airways to optimize resource allocation, personalize marketing efforts, and understand customer intent.

The analysis leverages the Random Forest classification algorithm to predict booking completion based on a variety of features related to the booking and customer behavior.

## Notebook Contents

* **`Predicting Customer Booking Behavior with Random Forest.ipynb`**: This notebook contains the complete end-to-end analysis, including:
    * Data loading and exploration.
    * Feature selection and preprocessing.
    * Implementation of the Random Forest classification algorithm using `scikit-learn`.
    * Model training and hyperparameter tuning using Grid Search.
    * Model evaluation using various metrics such as accuracy, precision, recall, F1-score, and cross-validation score.
    * Analysis of feature importance to understand which factors most influence booking completion.

## Model Details

The primary model used in this analysis is the **Random Forest** classification algorithm. Key parameters and performance metrics of the initial model and the optimized model are summarized below:

**Initial Random Forest Model:**

| Parameter            | Value |
| -------------------- | ----- |
| Algorithm            | Random Forest |
| Number of Estimators | 300   |
| Max Depth            | 30    |
| Min Samples Split    | 10    |
| Min Samples Leaf     | 4     |
| Accuracy             | 0.737 |
| Precision (Class 1)  | 0.5291 |
| Cross-Validation Score | 0.73702 |

**Optimized Random Forest Model (after Grid Search):**

| Parameter            | Best Value |
| -------------------- | ---------- |
| Max Depth            | 99         |
| Min Samples Leaf     | 3          |
| Min Samples Split    | 8          |
| Number of Estimators | 101        |
| Accuracy             | 0.8535     |
| Precision (Class 0)  | 0.86       |
| Precision (Class 1)  | 0.55       |
| Recall (Class 0)     | 0.99       |
| Recall (Class 1)     | 0.05       |
| F1-Score (Class 0)   | 0.92       |
| F1-Score (Class 1)   | 0.10       |
| Accuracy             | 0.85       |
| Macro Avg F1-Score   | 0.51       |
| Weighted Avg F1-Score| 0.80       |

**Feature Importance (Top Features):**

| Feature               | Importance Score |
| --------------------- | ---------------- |
| Purchase\_lead        | 0.145165         |
| Flight\_hour          | 0.120342         |
| Length\_of\_stay      | 0.109337         |
| Num\_passengers       | 0.046724         |
| Flight\_duration      | 0.036990         |

## Libraries Used

The following Python libraries were used in this analysis:

* **`pandas`**: For data manipulation and analysis.
* **`numpy`**: For numerical computations.
* **`scikit-learn` (`sklearn`)**: For machine learning algorithms (Random Forest), model selection (GridSearchCV, train\_test\_split, cross\_val\_score), and evaluation metrics (accuracy, precision, recall, f1\_score).
* **`matplotlib`** and **`seaborn`** (likely): For data visualization (though not explicitly mentioned in the provided snippets, they are standard for EDA and model result visualization).

## Further Enhancements (Based on Provided `Random Forest Classification algorithm.docx`)

The analysis could be further enhanced by considering the following steps:

1.  **Model Evaluation and Tuning:**
    * Performing thorough cross-validation to ensure model robustness.
    * Further exploring hyperparameter tuning using techniques like Random Search.

2.  **Feature Engineering:**
    * Creating interaction features between existing variables to capture more complex relationships.
    * Implementing feature selection techniques to identify the most impactful features and potentially simplify the model.

3.  **Model Comparison:**
    * Experimenting with other classification algorithms such as Gradient Boosting, XGBoost, or Logistic Regression to compare performance.
    * Exploring ensemble methods to potentially improve prediction accuracy.

4.  **Performance Metrics:**
    * Analyzing the ROC-AUC curve and other relevant metrics for a more comprehensive evaluation, especially considering the class imbalance potentially indicated by the precision and recall scores of the optimized model.

## Getting Started

To view the analysis, simply open the `Predicting Customer Booking Behavior with Random Forest.ipynb` file using Jupyter Notebook or JupyterLab. Ensure you have the necessary libraries installed (`pandas`, `numpy`, `scikit-learn`, and potentially `matplotlib`, `seaborn`).

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
