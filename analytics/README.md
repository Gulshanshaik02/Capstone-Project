# Module 2 — Analytics Pipeline

## Overview

This module implements an end-to-end analytics and machine-learning pipeline using the Titanic dataset.

The pipeline performs:

1. Dataset loading and profiling
2. Missing-value analysis
3. Data cleaning
4. Univariate analysis
5. Bivariate survival analysis
6. Correlation analysis
7. Multivariate visualization
8. Standardization check
9. Stratified train/test split
10. Preprocessing using ColumnTransformer
11. Logistic Regression
12. Decision Tree
13. Random Forest
14. Imbalance handling
15. Random Forest hyperparameter tuning
16. Fare regression
17. Model evaluation
18. Complete pipeline saving with Joblib

---

# 1. Dataset Loading

The Titanic dataset is loaded using:

```python
sns.load_dataset("titanic")

The dataset is loaded once and immediately saved as:

analytics/titanic.csv

This CSV acts as the offline fallback for future runs.

The modeling stages continue using the same cleaned dataset rather than independently loading the Titanic dataset again.

2. Dataset Profiling

The pipeline reports:

Dataset shape
Dataset information using df.info()
Descriptive statistics using df.describe()
Missing-value percentages

The missing-value report is saved under:

outputs/missing_values.csv
3. Missing-Value Handling

Missing values are handled according to the assignment threshold:

Less than 5% missing → affected rows are dropped.
5%–30% missing → values are imputed.
More than 30% missing → the column is removed when imputation is considered unreliable.

For numeric columns, median imputation is used.

For categorical columns, the value "Missing" can be used where appropriate.

The exact missing percentages are printed by the program before the cleaning strategy is applied.

4. Univariate Analysis

The pipeline analyzes:

Age
Fare

For both variables it produces:

Histogram
Box plot
IQR-based outlier count

The IQR rule is:

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

Fare statistics include:

Mean
Median
Mode

The mean/median/mode relationship is used to describe the distribution's skewness.

Charts are stored in:

outputs/charts/
5. Survival Analysis

Survival rates are calculated by:

Sex
Passenger class
Sex and passenger class together

These analyses help describe differences in survival across demographic and passenger-class groups.

Charts include:

survival_by_sex.png
survival_by_pclass.png
survival_by_sex_pclass.png
Interpretation

The survival analysis shows that survival was not distributed equally across passenger groups. Sex and passenger class both provide useful information for explaining differences in survival outcomes.

6. Correlation Analysis

The correlation matrix uses exactly these six columns:

survived
pclass
age
sibsp
parch
fare

The boolean columns adult_male and alone are excluded because they are derived/redundant features.

The correlation matrix is visualized using a Seaborn heatmap.

Output:

outputs/charts/correlation_heatmap.png

The two strongest correlations are identified by ranking the absolute values of the off-diagonal correlation coefficients.

7. Multivariate Data Story

The pipeline produces multiple charts to investigate relationships between passenger characteristics and survival.

Examples include:

Age vs survival
Fare vs survival
Age vs fare by sex and survival
Survival by sex
Survival by passenger class
Survival by sex and passenger class
Chart Interpretation
Survival by Sex

The chart compares survival rates between male and female passengers. It provides a clear view of the relationship between sex and survival.

Survival by Passenger Class

Passenger class is compared against survival rate. This helps investigate whether socioeconomic class was associated with different survival outcomes.

Age vs Survival

The age distribution is compared between survived and non-survived passengers. This helps identify differences in age distributions across the two outcome groups.

Fare vs Survival

Fare distributions are compared between passengers who survived and those who did not. Fare provides an additional indicator related to passenger class and ticket characteristics.

8. Standardization

Age and fare are standardized using the z-score:

z = (x - mean) / standard deviation

The pipeline prints the mean and standard deviation before and after standardization.

The transformed variables should have approximately:

Mean = 0
Standard deviation = 1

This standardization is an exploratory check and is separate from the modeling pipeline's train-only preprocessing.

9. Train/Test Split

The target variable is:

survived

A stratified train/test split is used.

Stratification is important because the survived and non-survived classes are not perfectly balanced. It helps preserve approximately the same class proportions in the training and testing datasets.

10. Model Preprocessing

The modeling pipeline uses:

ColumnTransformer

Numeric features use:

Median imputation
StandardScaler

Categorical features use:

Most-frequent imputation
OneHotEncoder

The preprocessing is fitted only on the training data and then applied to the test data.

This prevents test-set information from leaking into the training process.

11. Classification Models

Three classification models are trained using the same train/test split:

Logistic Regression

Used as a linear classification baseline.

Decision Tree

Used to model nonlinear relationships.

A tree visualization is generated using:

plot_tree()
Random Forest

Uses multiple decision trees to improve predictive robustness.

12. Model Evaluation

Each classifier is evaluated using:

Confusion Matrix
Accuracy
Precision
Recall
F1 Score
ROC Curve
ROC-AUC

The results are collected into a comparison table.

13. Class Imbalance

Three approaches are compared:

Baseline model
class_weight="balanced"
SMOTE

SMOTE is applied only to the training data to prevent data leakage.

The comparison focuses on:

Precision
Recall
F1 Score

The printed results are used to determine which strategy provides the most suitable balance for the task.

14. Random Forest Hyperparameter Tuning

GridSearchCV is used to tune the Random Forest.

The search includes:

n_estimators
max_depth
max_features

The tuned Random Forest uses:

oob_score=True

The best parameters and OOB score are reported.

15. Fare Regression

A multivariate linear regression model is used to predict:

fare

from the available passenger features.

The following metrics are reported:

MAE
RMSE
R²
Adjusted R²

A residual plot is also generated.

The residual plot is inspected for evidence of heteroscedasticity.

16. Model Comparison

Classification metrics are compared separately:

Model	Accuracy	Precision	Recall	F1	AUC
Logistic Regression	See output	See output	See output	See output	See output
Decision Tree	See output	See output	See output	See output	See output
Random Forest	See output	See output	See output	See output	See output

Regression metrics are reported separately:

Model	MAE	RMSE	R²	Adjusted R²
Linear Regression	See output	See output	See output	See output

Classification and regression metrics are not directly comparable because they measure different objectives.

The final classifier recommendation is based on the actual metric values produced by the pipeline.

17. Saved Model

The best complete machine-learning pipeline is saved using Joblib.

The saved object contains:

Preprocessing
    ↓
Imputation
    ↓
Encoding
    ↓
Scaling
    ↓
Final Estimator

This allows raw input data to be passed directly to the saved pipeline without manually preprocessing it first.

The pipeline is reloaded using:

joblib.load()

and tested on raw input.

18. Output Directory

The generated outputs are organized as:

analytics/
│
├── analytics.py
├── titanic.csv
├── README.md
│
└── outputs/
    ├── missing_values.csv
    │
    ├── charts/
    │   ├── age_histogram.png
    │   ├── age_boxplot.png
    │   ├── fare_histogram.png
    │   ├── fare_boxplot.png
    │   ├── survival_by_sex.png
    │   ├── survival_by_pclass.png
    │   ├── survival_by_sex_pclass.png
    │   ├── correlation_heatmap.png
    │   ├── age_vs_survival.png
    │   ├── fare_vs_survival.png
    │   └── age_fare_sex_survival.png
    │
    └── models/
19. How to Run

From the project root:

python analytics\analytics.py

The script creates the Titanic fallback dataset and analytical outputs automatically.

20. Requirements

Required Python packages include:

pandas
numpy
seaborn
matplotlib
scikit-learn
imbalanced-learn
joblib

Install them using:

pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
21. Completion

This module provides an end-to-end workflow from dataset loading and cleaning through exploratory analysis, predictive modeling, model evaluation, hyperparameter tuning, regression analysis, and deployment-ready model serialization.


### Step 3 — Save the README

If using Notepad:

**File → Save**

Then close Notepad.

### Step 4 — Check the Analytics folder

Run:

```powershell
Get-ChildItem analytics

You should now have:

analytics
│
├── analytics.py
├── README.md
├── titanic.csv
└── outputs
Step 5 — Important: run Git status

From the project root:

git status

You should see the Analytics files as new/modified files.

Then:

git add analytics
git status

Don't commit yet. First we should verify that the generated titanic.csv, charts, and model files actually exist and that the Analytics script completed successfully.

Your immediate next command

Run:

Get-ChildItem analytics -Recurse | Select-Object FullName