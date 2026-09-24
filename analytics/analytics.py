import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

for folder in [OUTPUT_DIR, CHART_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

RANDOM_STATE = 42
CSV_PATH = os.path.join(BASE_DIR, "titanic.csv")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_ohe():
    """Support both older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, filename),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def iqr_outlier_info(series):
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(
        ((series < lower) | (series > upper)).sum()
    )

    return count, q1, q3, lower, upper


def survival_rate(frame):
    if len(frame) == 0:
        return np.nan
    return float(frame["survived"].mean())


def build_preprocessor(numeric_features, categorical_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe())
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ], remainder="drop")


# ============================================================
# START
# ============================================================

print("=" * 80)
print("MODULE 2 - TITANIC ANALYTICS PIPELINE")
print("=" * 80)


# ============================================================
# 1. LOAD TITANIC DATASET ONCE
# ============================================================

print("\n[1] LOADING TITANIC DATASET")

try:
    df = sns.load_dataset("titanic")
    print("Dataset loaded using sns.load_dataset('titanic').")

except Exception as error:

    print("Online Titanic loading failed:")
    print(error)

    if not os.path.exists(CSV_PATH):
        raise RuntimeError(
            "Could not load Titanic from Seaborn and "
            "analytics/titanic.csv does not exist."
        )

    df = pd.read_csv(CSV_PATH)
    print("Using analytics/titanic.csv fallback.")

# Required offline fallback
df.to_csv(CSV_PATH, index=False)

print("Saved raw dataset:", CSV_PATH)


# ============================================================
# 2. PROFILE
# ============================================================

print("\n" + "=" * 80)
print("[2] DATASET PROFILE")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe(include="all").transpose())


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("[3] MISSING VALUE ANALYSIS")
print("=" * 80)

missing_percent = df.isnull().mean() * 100

missing_report = (
    missing_percent[missing_percent > 0]
    .sort_values(ascending=False)
)

print("\nMissing percentages:")
print(missing_report)

missing_report.rename(
    "missing_percent"
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_values.csv"
    )
)


# ============================================================
# 4. CLEANING
# ============================================================

print("\n" + "=" * 80)
print("[4] CLEANING")
print("=" * 80)

clean_df = df.copy()

cleaning_decisions = []

for column, percentage in missing_report.items():

    if percentage < 5:

        clean_df = clean_df.dropna(
            subset=[column]
        )

        strategy = "Dropped rows because missingness < 5%"

    elif percentage <= 30:

        if pd.api.types.is_numeric_dtype(
            clean_df[column]
        ):

            clean_df[column] = clean_df[column].fillna(
                clean_df[column].median()
            )

            strategy = (
                "Median imputation because "
                "missingness is between 5% and 30%"
            )

        else:

            clean_df[column] = clean_df[column].fillna(
                "Missing"
            )

            strategy = (
                "Encoded missing values as 'Missing' "
                "because missingness is between 5% and 30%"
            )

    else:

        clean_df = clean_df.drop(
            columns=[column]
        )

        strategy = (
            "Dropped column because "
            "missingness is greater than 30%"
        )

    cleaning_decisions.append({
        "column": column,
        "missing_percent": round(
            float(percentage),
            4
        ),
        "strategy": strategy
    })


cleaning_report = pd.DataFrame(
    cleaning_decisions
)

print("\nCleaning decisions:")
print(
    cleaning_report.to_string(
        index=False
    )
)

cleaning_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cleaning_decisions.csv"
    ),
    index=False
)

print("\nCleaned shape:")
print(clean_df.shape)


# ============================================================
# 5. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("[5] UNIVARIATE ANALYSIS")
print("=" * 80)

outlier_results = []

for column in ["age", "fare"]:

    count, q1, q3, lower, upper = (
        iqr_outlier_info(
            clean_df[column]
        )
    )

    outlier_results.append({
        "feature": column,
        "outlier_count": count,
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "lower_bound": lower,
        "upper_bound": upper
    })

    print(f"\n{column.upper()} IQR OUTLIERS")
    print("Outlier count:", count)
    print("Q1:", q1)
    print("Q3:", q3)
    print("Lower bound:", lower)
    print("Upper bound:", upper)

    # Histogram
    plt.figure(figsize=(8, 5))

    sns.histplot(
        clean_df[column],
        kde=True
    )

    plt.title(
        f"{column.title()} Distribution"
    )

    plt.xlabel(column)

    save_chart(
        f"{column}_histogram.png"
    )

    # Box plot
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=clean_df[column]
    )

    plt.title(
        f"{column.title()} Box Plot"
    )

    plt.xlabel(column)

    save_chart(
        f"{column}_boxplot.png"
    )


outlier_report = pd.DataFrame(
    outlier_results
)

outlier_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "iqr_outliers.csv"
    ),
    index=False
)

print("\nIQR report:")
print(
    outlier_report.to_string(
        index=False
    )
)


# Fare statistics

fare_mean = clean_df["fare"].mean()
fare_median = clean_df["fare"].median()
fare_mode = clean_df["fare"].mode().iloc[0]
fare_skew = clean_df["fare"].skew()

print("\nFARE STATISTICS")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)
print("Skewness:", fare_skew)

if fare_mean > fare_median > fare_mode:

    fare_conclusion = (
        "Fare is right-skewed because "
        "mean > median > mode."
    )

elif fare_mean < fare_median < fare_mode:

    fare_conclusion = (
        "Fare is left-skewed because "
        "mean < median < mode."
    )

else:

    fare_conclusion = (
        "Fare does not follow a strict "
        "mean/median/mode ordering."
    )

print("Conclusion:", fare_conclusion)


# ============================================================
# 6. BIVARIATE SURVIVAL ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("[6] BIVARIATE SURVIVAL ANALYSIS")
print("=" * 80)


# ------------------------------------------------------------
# Survival by sex using boolean masking
# ------------------------------------------------------------

sex_results = []

for sex in sorted(
    clean_df["sex"].dropna().unique()
):

    mask = (
        clean_df["sex"] == sex
    )

    subset = clean_df.loc[mask]

    sex_results.append({
        "sex": sex,
        "count": int(mask.sum()),
        "survival_rate": survival_rate(
            subset
        )
    })


sex_survival = pd.DataFrame(
    sex_results
)

print("\nSurvival by sex:")
print(
    sex_survival.to_string(
        index=False
    )
)

sex_survival.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "survival_by_sex.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Survival by pclass using boolean masking
# ------------------------------------------------------------

pclass_results = []

for pclass in sorted(
    clean_df["pclass"].dropna().unique()
):

    mask = (
        clean_df["pclass"] == pclass
    )

    subset = clean_df.loc[mask]

    pclass_results.append({
        "pclass": int(pclass),
        "count": int(mask.sum()),
        "survival_rate": survival_rate(
            subset
        )
    })


pclass_survival = pd.DataFrame(
    pclass_results
)

print("\nSurvival by pclass:")
print(
    pclass_survival.to_string(
        index=False
    )
)

pclass_survival.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "survival_by_pclass.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Survival by sex + pclass using & boolean masking
# ------------------------------------------------------------

sex_pclass_results = []

for sex in sorted(
    clean_df["sex"].dropna().unique()
):

    for pclass in sorted(
        clean_df["pclass"].dropna().unique()
    ):

        mask = (
            (clean_df["sex"] == sex)
            &
            (clean_df["pclass"] == pclass)
        )

        subset = clean_df.loc[mask]

        sex_pclass_results.append({
            "sex": sex,
            "pclass": int(pclass),
            "count": int(mask.sum()),
            "survival_rate": survival_rate(
                subset
            )
        })


sex_pclass_survival = pd.DataFrame(
    sex_pclass_results
)

print("\nSurvival by sex and pclass:")
print(
    sex_pclass_survival.to_string(
        index=False
    )
)

sex_pclass_survival.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "survival_by_sex_pclass.csv"
    ),
    index=False
)


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("[7] CORRELATION ANALYSIS")
print("=" * 80)

# EXACTLY the six required columns
correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = (
    clean_df[
        correlation_columns
    ].corr()
)

print("\nCorrelation matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlation_matrix.csv"
    )
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title(
    "Titanic Correlation Matrix"
)

save_chart(
    "correlation_heatmap.png"
)


# Find two strongest absolute correlations

correlation_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        value = correlation_matrix.iloc[
            i,
            j
        ]

        correlation_pairs.append({
            "feature_1": correlation_columns[i],
            "feature_2": correlation_columns[j],
            "correlation": value,
            "absolute_correlation": abs(value)
        })


correlation_pairs_df = (
    pd.DataFrame(
        correlation_pairs
    )
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)

print(
    "\nTwo strongest absolute correlations:"
)

print(
    correlation_pairs_df.head(2).to_string(
        index=False
    )
)

correlation_pairs_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correlation_pairs.csv"
    ),
    index=False
)


# ============================================================
# 8. MULTIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("[8] MULTIVARIATE ANALYSIS")
print("=" * 80)


# Chart 1: Survival by class and sex

plt.figure(figsize=(9, 6))

sns.barplot(
    data=clean_df,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None
)

plt.title(
    "Survival Rate by Passenger Class and Sex"
)

plt.ylabel(
    "Survival Rate"
)

save_chart(
    "multivariate_survival_pclass_sex.png"
)


# Chart 2: Fare by class and sex

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=clean_df,
    x="pclass",
    y="fare",
    hue="sex"
)

plt.title(
    "Fare by Passenger Class and Sex"
)

save_chart(
    "multivariate_fare_pclass_sex.png"
)


# Chart 3: Age vs fare by survival and sex

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=clean_df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title(
    "Age vs Fare by Survival and Sex"
)

save_chart(
    "multivariate_age_fare_survival_sex.png"
)


# Chart 4: Age by class and survival

plt.figure(figsize=(9, 6))

sns.violinplot(
    data=clean_df,
    x="pclass",
    y="age",
    hue="survived",
    split=True
)

plt.title(
    "Age Distribution by Class and Survival"
)

save_chart(
    "multivariate_age_pclass_survival.png"
)


# Chart 5: Family size vs survival

family_df = clean_df.copy()

family_df["family_size"] = (
    family_df["sibsp"]
    + family_df["parch"]
    + 1
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=family_df,
    x="family_size",
    y="survived",
    errorbar=None
)

plt.title(
    "Survival Rate by Family Size"
)

plt.ylabel(
    "Survival Rate"
)

save_chart(
    "multivariate_family_size_survival.png"
)


# Required written interpretations

multivariate_interpretations = """
MULTIVARIATE CHART INTERPRETATIONS

1. Survival Rate by Passenger Class and Sex:
This chart compares survival rates jointly across passenger class and sex.
It shows that survival outcomes varied across combinations of class and sex,
rather than being explained by only one variable.

2. Fare by Passenger Class and Sex:
Fare distributions differ across passenger classes, with higher classes
generally showing higher fare values. The sex split also reveals variation
within each passenger class.

3. Age vs Fare by Survival and Sex:
The scatter plot shows how age and fare are distributed together while
separating observations by survival and sex. Fare values are concentrated
at lower levels, while observations span a broad range of ages.

4. Age Distribution by Class and Survival:
Age distributions differ across passenger classes and survival groups.
This visualization helps show the joint relationship between passenger
class, age, and survival.

5. Survival Rate by Family Size:
Survival rates vary across family-size groups. Very small and larger family
groups can display different survival patterns, showing that family size
provides additional context beyond class and sex alone.
""".strip()

print("\n" + multivariate_interpretations)

with open(
    os.path.join(
        OUTPUT_DIR,
        "multivariate_interpretations.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        multivariate_interpretations
    )


# ============================================================
# 9. EXPLORATORY Z-SCORE STANDARDIZATION
# ============================================================

print("\n" + "=" * 80)
print("[9] EXPLORATORY STANDARDIZATION")
print("=" * 80)

standardization_results = []

for column in ["age", "fare"]:

    original_mean = clean_df[
        column
    ].mean()

    original_std = clean_df[
        column
    ].std()

    standardized = (
        clean_df[column]
        - original_mean
    ) / original_std

    standardization_results.append({
        "feature": column,
        "before_mean": original_mean,
        "before_std": original_std,
        "after_mean": standardized.mean(),
        "after_std": standardized.std()
    })


standardization_report = pd.DataFrame(
    standardization_results
)

print(
    standardization_report.to_string(
        index=False
    )
)

standardization_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "standardization_check.csv"
    ),
    index=False
)


# ============================================================
# 10. CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("[10] CLASSIFICATION MODELING")
print("=" * 80)

model_df = clean_df.copy()

# survived is target.
# alive is directly derived from survived, so it is excluded.
classifier_drop = [
    column
    for column in ["survived", "alive"]
    if column in model_df.columns
]

X = model_df.drop(
    columns=classifier_drop
)

y = model_df["survived"]


# Stratified split BEFORE preprocessing

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print("\nClass distribution:")
print(
    y.value_counts()
)

print("\nClass proportions:")
print(
    y.value_counts(
        normalize=True
    )
)

print(
    "\nStratification is used so the "
    "survived/not-survived proportions "
    "remain similar in train and test."
)


numeric_features = (
    X_train
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

categorical_features = (
    X_train
    .select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# Three classifiers
# ------------------------------------------------------------

classifiers = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        )
}


classification_results = []
roc_results = {}

fitted_pipelines = {}


for model_name, estimator in classifiers.items():

    pipeline = Pipeline([

        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features
            )
        ),

        (
            "classifier",
            estimator
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    fitted_pipelines[
        model_name
    ] = pipeline

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    cm = confusion_matrix(
        y_test,
        predictions
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    classification_results.append({

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": auc
    })

    print("\n" + "-" * 60)

    print(model_name)

    print("\nConfusion Matrix:")
    print(cm)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    # Confusion matrix chart

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    save_chart(filename)

    # ROC data

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_results[
        model_name
    ] = (
        fpr,
        tpr,
        auc
    )


classification_table = pd.DataFrame(
    classification_results
)

print("\nCLASSIFICATION COMPARISON")
print(
    classification_table.to_string(
        index=False
    )
)

classification_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_results.csv"
    ),
    index=False
)


# ROC comparison

plt.figure(figsize=(8, 6))

for model_name, (
    fpr,
    tpr,
    auc
) in roc_results.items():

    plt.plot(
        fpr,
        tpr,
        label=(
            f"{model_name} "
            f"AUC={auc:.3f}"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Classifier ROC Curves"
)

plt.legend()

save_chart(
    "classifier_roc_curves.png"
)


# ============================================================
# 11. DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("[11] DECISION TREE VISUALIZATION")
print("=" * 80)

tree_pipeline = fitted_pipelines[
    "Decision Tree"
]

tree_preprocessor = (
    tree_pipeline
    .named_steps[
        "preprocessor"
    ]
)

tree_model = (
    tree_pipeline
    .named_steps[
        "classifier"
    ]
)

tree_feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)

plt.figure(
    figsize=(24, 12)
)

plot_tree(
    tree_model,
    feature_names=tree_feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=False,
    max_depth=4,
    fontsize=7
)

plt.title(
    "Decision Tree Classifier"
)

save_chart(
    "decision_tree.png"
)


# ============================================================
# 12. CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 80)
print("[12] CLASS IMBALANCE COMPARISON")
print("=" * 80)


def evaluate_rf(
    strategy_name,
    class_weight=None
):

    pipeline = Pipeline([

        (
            "preprocessor",
            build_preprocessor(
                numeric_features,
                categorical_features
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight=class_weight,
                random_state=RANDOM_STATE
            )
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    return {

        "strategy": strategy_name,

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }


imbalance_results = []

imbalance_results.append(
    evaluate_rf(
        "Baseline"
    )
)

imbalance_results.append(
    evaluate_rf(
        "class_weight=balanced",
        class_weight="balanced"
    )
)


# SMOTE is applied ONLY to transformed training data.

smote_preprocessor = build_preprocessor(
    numeric_features,
    categorical_features
)

X_train_transformed = (
    smote_preprocessor.fit_transform(
        X_train
    )
)

X_test_transformed = (
    smote_preprocessor.transform(
        X_test
    )
)

smote = SMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_transformed,
        y_train
    )
)

smote_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

smote_predictions = (
    smote_model.predict(
        X_test_transformed
    )
)

imbalance_results.append({

    "strategy": "SMOTE",

    "precision": precision_score(
        y_test,
        smote_predictions,
        zero_division=0
    ),

    "recall": recall_score(
        y_test,
        smote_predictions,
        zero_division=0
    ),

    "f1": f1_score(
        y_test,
        smote_predictions,
        zero_division=0
    )
})


imbalance_table = pd.DataFrame(
    imbalance_results
)

print(
    imbalance_table.to_string(
        index=False
    )
)

imbalance_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_comparison.csv"
    ),
    index=False
)

best_imbalance = (
    imbalance_table
    .sort_values(
        "f1",
        ascending=False
    )
    .iloc[0]
)

imbalance_conclusion = (
    f"The tested {best_imbalance['strategy']} "
    f"strategy produced the highest test F1 "
    f"of {best_imbalance['f1']:.4f}. "
    f"Precision and recall should also be considered "
    f"because the preferred balance depends on the "
    f"cost of false positives and false negatives."
)

print("\nImbalance conclusion:")
print(imbalance_conclusion)


# ============================================================
# 13. RANDOM FOREST GRID SEARCH + OOB
# ============================================================

print("\n" + "=" * 80)
print("[13] RANDOM FOREST GRID SEARCH")
print("=" * 80)


rf_pipeline = Pipeline([

    (
        "preprocessor",
        build_preprocessor(
            numeric_features,
            categorical_features
        )
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            oob_score=True
        )
    )
])


parameter_grid = {

    "classifier__n_estimators": [
        200,
        300
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(

    rf_pipeline,

    param_grid=parameter_grid,

    cv=5,

    scoring="f1",

    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


best_rf_pipeline = (
    grid_search.best_estimator_
)

best_rf_model = (
    best_rf_pipeline
    .named_steps[
        "classifier"
    ]
)

print("\nBest parameters:")
print(
    grid_search.best_params_
)

print(
    "\nBest CV F1:",
    grid_search.best_score_
)

print(
    "OOB score:",
    best_rf_model.oob_score_
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_tuning.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Random Forest Grid Search\n\n"
    )

    file.write(
        f"Best parameters: "
        f"{grid_search.best_params_}\n"
    )

    file.write(
        f"Best CV F1: "
        f"{grid_search.best_score_:.6f}\n"
    )

    file.write(
        f"OOB score: "
        f"{best_rf_model.oob_score_:.6f}\n"
    )


# Evaluate tuned RF

tuned_predictions = (
    best_rf_pipeline.predict(
        X_test
    )
)

tuned_probabilities = (
    best_rf_pipeline
    .predict_proba(
        X_test
    )[:, 1]
)

tuned_rf_metrics = {

    "model":
        "Tuned Random Forest",

    "accuracy":
        accuracy_score(
            y_test,
            tuned_predictions
        ),

    "precision":
        precision_score(
            y_test,
            tuned_predictions,
            zero_division=0
        ),

    "recall":
        recall_score(
            y_test,
            tuned_predictions,
            zero_division=0
        ),

    "f1":
        f1_score(
            y_test,
            tuned_predictions,
            zero_division=0
        ),

    "roc_auc":
        roc_auc_score(
            y_test,
            tuned_probabilities
        )
}


classification_table = pd.concat(
    [
        classification_table,
        pd.DataFrame(
            [tuned_rf_metrics]
        )
    ],
    ignore_index=True
)

classification_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_results.csv"
    ),
    index=False
)


# ============================================================
# 14. FARE REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("[14] MULTIVARIATE FARE REGRESSION")
print("=" * 80)

regression_df = clean_df.copy()

# Predict fare from other available features.
# survived/alive are excluded because they are target-related fields.

regression_drop = [
    column
    for column in [
        "fare",
        "survived",
        "alive"
    ]
    if column in regression_df.columns
]

X_regression = regression_df.drop(
    columns=regression_drop
)

y_regression = regression_df[
    "fare"
]


Xr_train, Xr_test, yr_train, yr_test = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=RANDOM_STATE
    )
)


regression_numeric = (
    Xr_train
    .select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

regression_categorical = (
    Xr_train
    .select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)


regression_pipeline = Pipeline([

    (
        "preprocessor",
        build_preprocessor(
            regression_numeric,
            regression_categorical
        )
    ),

    (
        "regressor",
        LinearRegression()
    )
])


regression_pipeline.fit(
    Xr_train,
    yr_train
)

fare_predictions = (
    regression_pipeline.predict(
        Xr_test
    )
)


mae = mean_absolute_error(
    yr_test,
    fare_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        yr_test,
        fare_predictions
    )
)

r2 = r2_score(
    yr_test,
    fare_predictions
)


# Adjusted R2

number_of_observations = len(
    yr_test
)

number_of_features = len(
    regression_pipeline
    .named_steps[
        "preprocessor"
    ]
    .get_feature_names_out()
)

if number_of_observations > (
    number_of_features + 1
):

    adjusted_r2 = (
        1
        -
        (
            (1 - r2)
            *
            (
                (number_of_observations - 1)
                /
                (
                    number_of_observations
                    -
                    number_of_features
                    -
                    1
                )
            )
        )
    )

else:

    adjusted_r2 = np.nan


# Residuals

residuals = (
    yr_test.to_numpy()
    -
    fare_predictions
)


# Simple heteroscedasticity diagnostic:
# correlation between predicted fare and absolute residual.

if len(residuals) > 1:

    residual_correlation = np.corrcoef(
        fare_predictions,
        np.abs(residuals)
    )[0, 1]

else:

    residual_correlation = np.nan


if np.isnan(
    residual_correlation
):

    heteroscedasticity_conclusion = (
        "Heteroscedasticity could not be "
        "assessed reliably."
    )

elif abs(
    residual_correlation
) >= 0.30:

    heteroscedasticity_conclusion = (
        "The residual spread changes with "
        "predicted fare, suggesting evidence "
        "of heteroscedasticity."
    )

else:

    heteroscedasticity_conclusion = (
        "There is no strong linear association "
        "between predicted fare and absolute "
        "residuals, so this diagnostic provides "
        "limited evidence of heteroscedasticity."
    )


print("\nRegression metrics:")

print(
    f"MAE        : {mae:.4f}"
)

print(
    f"RMSE       : {rmse:.4f}"
)

print(
    f"R2         : {r2:.4f}"
)

print(
    f"Adjusted R2: {adjusted_r2:.4f}"
)

print(
    "\nResidual correlation:",
    residual_correlation
)

print(
    "Heteroscedasticity conclusion:",
    heteroscedasticity_conclusion
)


# Residual plot

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    fare_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Fare Regression Residual Plot"
)

save_chart(
    "fare_regression_residuals.png"
)


regression_table = pd.DataFrame([{

    "model":
        "Multivariate Linear Regression",

    "MAE":
        mae,

    "RMSE":
        rmse,

    "R2":
        r2,

    "Adjusted_R2":
        adjusted_r2

}])


print("\nRegression comparison:")
print(
    regression_table.to_string(
        index=False
    )
)

regression_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_results.csv"
    ),
    index=False
)


# ============================================================
# 15. SAVE + RELOAD COMPLETE PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("[15] SAVE AND RELOAD FINAL PIPELINE")
print("=" * 80)


# Complete fitted preprocessing + estimator pipeline

model_path = os.path.join(
    MODEL_DIR,
    "titanic_random_forest_pipeline.joblib"
)

joblib.dump(
    best_rf_pipeline,
    model_path
)

print(
    "Saved:",
    model_path
)


# Reload

loaded_pipeline = joblib.load(
    model_path
)

print(
    "Pipeline reloaded successfully."
)


# Demonstrate prediction using raw/unprocessed features

raw_sample = X_test.iloc[
    [0]
].copy()

raw_prediction = (
    loaded_pipeline.predict(
        raw_sample
    )[0]
)

raw_probability = (
    loaded_pipeline
    .predict_proba(
        raw_sample
    )[0, 1]
)

print(
    "\nRaw input sample prediction:",
    int(raw_prediction)
)

print(
    "Predicted survival probability:",
    round(
        float(raw_probability),
        4
    )
)


# ============================================================
# 16. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("[16] FINAL MODEL COMPARISON")
print("=" * 80)

print("\nCLASSIFIER METRICS")
print(
    classification_table.to_string(
        index=False
    )
)

print("\nREGRESSION METRICS")
print(
    regression_table.to_string(
        index=False
    )
)


classification_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_classification_comparison.csv"
    ),
    index=False
)


# ============================================================
# 17. FINAL WRITTEN RECOMMENDATION
# ============================================================

final_recommendation = f"""
FINAL RECOMMENDATION

1. The classification models should be compared using accuracy, precision,
recall, F1, and ROC-AUC. F1 is particularly useful when both false positives
and false negatives matter.

2. The Random Forest was tuned with GridSearchCV and evaluated using its
out-of-bag score, and the complete fitted preprocessing + Random Forest
pipeline was saved as the final reusable classifier artifact.

3. The class-imbalance experiment compared a baseline Random Forest,
class_weight='balanced', and SMOTE applied only to the training data.
The strongest test F1 among these strategies was produced by:
{best_imbalance['strategy']} with F1={best_imbalance['f1']:.4f}.

4. Fare prediction is a separate regression task. It should be evaluated
using MAE, RMSE, R2, and Adjusted R2 rather than classification metrics.

5. The fare residual diagnostic concluded:
{heteroscedasticity_conclusion}
""".strip()


print(
    "\n" + final_recommendation
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "final_recommendation.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        final_recommendation
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("MODULE 2 COMPLETED")
print("=" * 80)

print(
    "\nTitanic CSV:",
    CSV_PATH
)

print(
    "Charts:",
    CHART_DIR
)

print(
    "Models:",
    MODEL_DIR
)

print(
    "Reports:",
    OUTPUT_DIR
)

print(
    "\nAll required Module 2 processing completed."
)
