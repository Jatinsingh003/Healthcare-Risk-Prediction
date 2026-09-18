# ============================================================
# HEALTHCARE PATIENT RISK ANALYSIS & PREDICTION
# ============================================================
#
# Project:
# Healthcare Patient Risk Prediction & Laboratory Analysis
#
# Technologies:
# Python
# Pandas
# Matplotlib
# Scikit-learn
# Joblib
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

import joblib


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# Get the folder where this Python file is located
BASE_PATH = Path(__file__).parent

# Create folders for outputs and models
OUTPUT_PATH = BASE_PATH / "outputs"
MODEL_PATH = BASE_PATH / "models"

OUTPUT_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)


print("=" * 70)
print("HEALTHCARE PATIENT RISK ANALYSIS & PREDICTION")
print("=" * 70)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[1/15] Loading datasets...")

patients = pd.read_csv(BASE_PATH / "patients.csv")
diagnosis = pd.read_csv(BASE_PATH / "diagnosis.csv")
outcomes = pd.read_csv(BASE_PATH / "outcomes.csv")
labs = pd.read_csv(BASE_PATH / "labs.csv")

print("Patients loaded :", len(patients))
print("Diagnosis loaded:", len(diagnosis))
print("Outcomes loaded :", len(outcomes))
print("Labs loaded     :", len(labs))

# Keep only the original patient table columns.
# This prevents previously generated Python columns
# from interfering with a new run.

# ============================================================
# CLEAN RAW PATIENT DATA
# ============================================================

raw_patient_columns = [
    "patient_id",
    "name",
    "age",
    "gender",
    "diagnosis_id",
    "admission_date",
    "discharge_date",
    "outcome_id",
    "treatment_cost"
]

patients = patients[
    [col for col in raw_patient_columns if col in patients.columns]
].copy()

# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

print("\n[2/15] Cleaning column names...")


def clean_column_names(df):
    """
    Removes unnecessary spaces and standardizes column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


patients = clean_column_names(patients)
diagnosis = clean_column_names(diagnosis)
outcomes = clean_column_names(outcomes)
labs = clean_column_names(labs)


# ------------------------------------------------------------
# Correct common spelling mistakes
# ------------------------------------------------------------

# Patients table
patients = patients.rename(
    columns={
        "diagnoses_id": "diagnosis_id",
        "treatement_cost": "treatment_cost",
        "treatment_cost": "treatment_cost"
    }
)

# Diagnosis table
diagnosis = diagnosis.rename(
    columns={
        "diagnoses_id": "diagnosis_id",
        "diagnoses_name": "diagnosis_name"
    }
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[3/15] Checking dataset structure...")


required_patient_columns = [
    "patient_id",
    "name",
    "age",
    "gender",
    "diagnosis_id",
    "admission_date",
    "discharge_date",
    "outcome_id",
    "treatment_cost"
]

required_diagnosis_columns = [
    "diagnosis_id",
    "diagnosis_name"
]

required_outcome_columns = [
    "outcome_id",
    "outcome_name"
]

required_lab_columns = [
    "lab_id",
    "patient_id",
    "test_name",
    "result",
    "normal_range"
]


def check_columns(df, required_columns, table_name):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(f"\nERROR in {table_name} table.")

        print("Missing columns:")
        for column in missing:
            print(" -", column)

        print("\nAvailable columns:")
        print(df.columns.tolist())

        raise ValueError(
            f"Missing required columns in {table_name}: {missing}"
        )


check_columns(
    patients,
    required_patient_columns,
    "patients"
)

check_columns(
    diagnosis,
    required_diagnosis_columns,
    "diagnosis"
)

check_columns(
    outcomes,
    required_outcome_columns,
    "outcomes"
)

check_columns(
    labs,
    required_lab_columns,
    "labs"
)

print("All required columns are available.")


# ============================================================
# 6. DATA TYPE CONVERSION
# ============================================================

print("\n[4/15] Converting data types...")


patients["admission_date"] = pd.to_datetime(
    patients["admission_date"],
    errors="coerce"
)

patients["discharge_date"] = pd.to_datetime(
    patients["discharge_date"],
    errors="coerce"
)

patients["age"] = pd.to_numeric(
    patients["age"],
    errors="coerce"
)

patients["treatment_cost"] = pd.to_numeric(
    patients["treatment_cost"],
    errors="coerce"
)

labs["result"] = pd.to_numeric(
    labs["result"],
    errors="coerce"
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print("\n[5/15] Checking missing values...")


print("\nMissing values in patients:")
print(patients.isnull().sum())

print("\nMissing values in diagnosis:")
print(diagnosis.isnull().sum())

print("\nMissing values in outcomes:")
print(outcomes.isnull().sum())

print("\nMissing values in labs:")
print(labs.isnull().sum())


# ============================================================
# 8. MERGE PATIENT + DIAGNOSIS + OUTCOME
# ============================================================

print("\n[6/15] Combining patient, diagnosis and outcome data...")


patients = patients.merge(
    diagnosis[
        [
            "diagnosis_id",
            "diagnosis_name"
        ]
    ],
    on="diagnosis_id",
    how="left"
)


patients = patients.merge(
    outcomes[
        [
            "outcome_id",
            "outcome_name"
        ]
    ],
    on="outcome_id",
    how="left"
)


print("Patient dataset after merging:")
print(patients.head())


# ============================================================
# 9. CALCULATE HOSPITAL STAY
# ============================================================

print("\n[7/15] Calculating hospital stay...")


patients["length_of_stay"] = (
    patients["discharge_date"]
    - patients["admission_date"]
).dt.days


# Prevent negative values if any bad dates exist
patients.loc[
    patients["length_of_stay"] < 0,
    "length_of_stay"
] = np.nan


# ============================================================
# 10. ABNORMAL LAB DETECTION
# ============================================================

import re


def check_abnormal_result(result, normal_range):

    if pd.isna(result) or pd.isna(normal_range):
        return np.nan

    try:
        result = float(result)
    except (ValueError, TypeError):
        return "Not Evaluated"

    normal_range = str(normal_range).strip()

    # Remove common units and keep numbers/operators
    normal_range_clean = normal_range.replace(",", "")

    # --------------------------------------------------------
    # Range: 70-100, 70 - 100, 70–100
    # --------------------------------------------------------

    range_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)",
        normal_range_clean
    )

    if range_match:

        lower = float(range_match.group(1))
        upper = float(range_match.group(2))

        if lower <= result <= upper:
            return "Normal"
        else:
            return "Abnormal"

    # --------------------------------------------------------
    # Less than: <5
    # --------------------------------------------------------

    less_match = re.search(
        r"<\s*(\d+(?:\.\d+)?)",
        normal_range_clean
    )

    if less_match:

        limit = float(less_match.group(1))

        if result < limit:
            return "Normal"
        else:
            return "Abnormal"

    # --------------------------------------------------------
    # Greater than: >60
    # --------------------------------------------------------

    greater_match = re.search(
        r">\s*(\d+(?:\.\d+)?)",
        normal_range_clean
    )

    if greater_match:

        limit = float(greater_match.group(1))

        if result > limit:
            return "Normal"
        else:
            return "Abnormal"

    # --------------------------------------------------------
    # Could not understand the normal range
    # --------------------------------------------------------

    return "Not Evaluated"

labs["lab_status"] = labs.apply(
    lambda row: check_abnormal_result(
        row["result"],
        row["normal_range"]
    ),
    axis=1
)


# ============================================================
# 11. CALCULATE ABNORMAL LAB COUNT PER PATIENT
# ============================================================

print("\n[9/15] Calculating abnormal lab count per patient...")


abnormal_lab_summary = (

    labs[
        labs["lab_status"] == "Abnormal"
    ]

    .groupby("patient_id")

    .size()

    .reset_index(
        name="abnormal_lab_count"
    )
)


patients = patients.merge(
    abnormal_lab_summary,
    on="patient_id",
    how="left"
)


patients["abnormal_lab_count"] = (
    patients["abnormal_lab_count"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 12. CREATE LAB-BASED FEATURES
# ============================================================

print("\n[10/15] Creating laboratory features...")


lab_summary = (

    labs

    .groupby("patient_id")

    .agg(
        total_lab_tests=(
            "lab_id",
            "count"
        ),

        average_lab_result=(
            "result",
            "mean"
        ),

        maximum_lab_result=(
            "result",
            "max"
        ),

        minimum_lab_result=(
            "result",
            "min"
        )
    )

    .reset_index()
)


patients = patients.merge(
    lab_summary,
    on="patient_id",
    how="left"
)


# ============================================================
# 13. CREATE RISK TARGET
# ============================================================

print("\n[11/15] Creating risk target...")


# Project-level analytical classification:
#
# 0 = Lower Risk
# 1 = Higher Risk
#
# This is NOT a clinical diagnosis.
#
# It is based on the synthetic treatment outcomes
# in this project.

patients["risk_target"] = (
    patients["outcome_name"]
    .map(
        {
            "Recovered": 0,
            "Improved": 0,
            "Stable": 0,
            "Referred": 1,
            "Deceased": 1
        }
    )
)


# Remove rows where risk target could not be created
patients = patients.dropna(
    subset=["risk_target"]
)

patients["risk_target"] = (
    patients["risk_target"]
    .astype(int)
)


print("\nRisk target distribution:")

print(
    patients["risk_target"]
    .value_counts()
    .rename(
        index={
            0: "Lower Risk",
            1: "Higher Risk"
        }
    )
)


# ============================================================
# 14. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n[12/15] Performing exploratory analysis...")


# ------------------------------------------------------------
# Risk distribution
# ------------------------------------------------------------

risk_distribution = (
    patients["risk_target"]
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(8, 5))

risk_distribution.plot(
    kind="bar"
)

plt.title(
    "Patient Risk Distribution"
)

plt.xlabel(
    "Risk Category"
)

plt.ylabel(
    "Number of Patients"
)

plt.xticks(
    [0, 1],
    [
        "Lower Risk",
        "Higher Risk"
    ],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "risk_distribution.png"
)

plt.show()


# ------------------------------------------------------------
# Abnormal labs by diagnosis
# ------------------------------------------------------------

diagnosis_analysis = (

    patients

    .groupby("diagnosis_name")

    .agg(

        patients=(
            "patient_id",
            "count"
        ),

        average_abnormal_labs=(
            "abnormal_lab_count",
            "mean"
        ),

        average_treatment_cost=(
            "treatment_cost",
            "mean"
        ),

        average_length_of_stay=(
            "length_of_stay",
            "mean"
        )
    )

    .sort_values(
        "average_abnormal_labs",
        ascending=False
    )
)


print("\nDiagnosis Analysis:")
print(diagnosis_analysis)


diagnosis_analysis[
    "average_abnormal_labs"
].sort_values().plot(
    kind="barh",
    figsize=(9, 6)
)

plt.title(
    "Average Abnormal Lab Results by Diagnosis"
)

plt.xlabel(
    "Average Abnormal Lab Results"
)

plt.ylabel(
    "Diagnosis"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "abnormal_labs_by_diagnosis.png"
)

plt.show()


# ============================================================
# 15. PREPARE MACHINE LEARNING DATA
# ============================================================

print("\n[13/15] Preparing machine learning dataset...")


features = [

    "age",

    "gender",

    "diagnosis_name",

    "treatment_cost",

    "length_of_stay",

    "abnormal_lab_count",

    "total_lab_tests",

    "average_lab_result",

    "maximum_lab_result",

    "minimum_lab_result"
]


X = patients[features].copy()

y = patients["risk_target"].copy()


# ------------------------------------------------------------
# Fill missing numerical values
# ------------------------------------------------------------

numerical_features = [

    "age",

    "treatment_cost",

    "length_of_stay",

    "abnormal_lab_count",

    "total_lab_tests",

    "average_lab_result",

    "maximum_lab_result",

    "minimum_lab_result"
]


categorical_features = [

    "gender",

    "diagnosis_name"
]


for column in numerical_features:

    X[column] = X[column].fillna(
        X[column].median()
    )


for column in categorical_features:

    X[column] = X[column].fillna(
        "Unknown"
    )


# ============================================================
# 16. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting data into training and testing sets...")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 17. PREPROCESSING
# ============================================================

print("\nCreating preprocessing pipeline...")


preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )

    ],

    remainder="passthrough"
)


# ============================================================
# 18. RANDOM FOREST MODEL
# ============================================================

print("\nCreating Random Forest model...")


model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced"
)


# ============================================================
# 19. COMPLETE ML PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ============================================================
# 20. TRAIN MODEL
# ============================================================

print("\nTraining machine learning model...")

pipeline.fit(
    X_train,
    y_train
)

print(
    "Model training completed successfully."
)


# ============================================================
# 21. MAKE TEST PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 22. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


# Accuracy

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy: {accuracy:.2%}"
)


# Classification report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Lower Risk",
            "Higher Risk"
        ]
    )
)


# ROC-AUC

try:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print(
        f"ROC-AUC Score: {roc_auc:.3f}"
    )

except ValueError:

    print(
        "ROC-AUC could not be calculated."
    )


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Lower Risk",
        "Higher Risk"
    ]
)


display.plot()

plt.title(
    "Confusion Matrix - Healthcare Risk Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "confusion_matrix.png"
)

plt.show()


# ============================================================
# 24. PREDICT RISK FOR ALL PATIENTS
# ============================================================

print("\nGenerating patient risk predictions...")


patients["predicted_risk"] = (
    pipeline.predict(
        X
    )
)


patients["predicted_risk_label"] = (
    patients["predicted_risk"]
    .map(
        {
            0: "Lower Risk",
            1: "Higher Risk"
        }
    )
)


patients["risk_probability"] = (
    pipeline.predict_proba(
        X
    )[:, 1]
)


patients["risk_probability_percent"] = (
    patients["risk_probability"] * 100
).round(2)


# ============================================================
# 25. CREATE FINAL PATIENT RISK REPORT
# ============================================================

print("\nCreating patient risk report...")


risk_report = patients[
    [

        "patient_id",

        "name",

        "age",

        "gender",

        "diagnosis_name",

        "outcome_name",

        "abnormal_lab_count",

        "total_lab_tests",

        "length_of_stay",

        "treatment_cost",

        "predicted_risk_label",

        "risk_probability_percent"

    ]
].sort_values(

    "risk_probability_percent",

    ascending=False
)


print("\nTop 20 patients by model risk probability:")

print(
    risk_report.head(20).to_string(
        index=False
    )
)


# ============================================================
# 26. HIGHER-RISK PATIENTS
# ============================================================

higher_risk_patients = risk_report[
    risk_report["predicted_risk_label"]
    == "Higher Risk"
]


print(
    "\nNumber of predicted higher-risk patients:",
    len(higher_risk_patients)
)


# ============================================================
# 27. ABNORMAL LAB REPORT
# ============================================================

abnormal_lab_report = (

    labs[
        labs["lab_status"] == "Abnormal"
    ]

    .merge(

        patients[
            [
                "patient_id",
                "name",
                "diagnosis_name"
            ]
        ],

        on="patient_id",

        how="left"
    )

)


abnormal_lab_report = abnormal_lab_report[
    [
        "patient_id",
        "name",
        "diagnosis_name",
        "test_name",
        "result",
        "normal_range",
        "lab_status"
    ]
]


# ============================================================
# 28. SAVE OUTPUT FILES
# ============================================================

print("\nSaving analysis files...")


# Complete patient dataset

patients.to_csv(
    OUTPUT_PATH / "processed_patient_data.csv",
    index=False
)


# Risk report

risk_report.to_csv(
    OUTPUT_PATH / "patient_risk_report.csv",
    index=False
)


# Higher-risk patients

higher_risk_patients.to_csv(
    OUTPUT_PATH / "higher_risk_patients.csv",
    index=False
)


# Abnormal laboratory results

abnormal_lab_report.to_csv(
    OUTPUT_PATH / "abnormal_lab_results.csv",
    index=False
)


# Diagnosis analysis

diagnosis_analysis.to_csv(
    OUTPUT_PATH / "diagnosis_analysis.csv"
)


# ============================================================
# 29. FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")


trained_model = (
    pipeline
    .named_steps["model"]
)


feature_names = (

    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()

)


feature_importance = pd.DataFrame(

    {

        "feature":
        feature_names,

        "importance":
        trained_model.feature_importances_

    }

)


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    "\nTop 15 important features:"
)

print(
    feature_importance
    .head(15)
    .to_string(
        index=False
    )
)


feature_importance.to_csv(

    OUTPUT_PATH /
    "feature_importance.csv",

    index=False
)


# ------------------------------------------------------------
# Feature importance chart
# ------------------------------------------------------------

feature_importance.head(15).sort_values(
    "importance"
).plot(

    x="feature",

    y="importance",

    kind="barh",

    legend=False,

    figsize=(10, 7)
)


plt.title(
    "Top Features Used by Risk Model"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_PATH /
    "feature_importance.png"
)


plt.show()


# ============================================================
# 30. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")


joblib.dump(

    pipeline,

    MODEL_PATH /
    "healthcare_risk_model.pkl"

)


print(
    "Model saved successfully."
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nTotal patients analyzed:",
    len(patients)
)

print(
    "Total laboratory records:",
    len(labs)
)

print(
    "Abnormal laboratory records:",
    len(abnormal_lab_report)
)

print(
    "Predicted higher-risk patients:",
    len(higher_risk_patients)
)


print(
    "\nOutput folder:"
)

print(
    OUTPUT_PATH
)


print(
    "\nModel folder:"
)

print(
    MODEL_PATH
)


print("\nGenerated files:")

print(
    "1. processed_patient_data.csv"
)

print(
    "2. patient_risk_report.csv"
)

print(
    "3. higher_risk_patients.csv"
)

print(
    "4. abnormal_lab_results.csv"
)

print(
    "5. diagnosis_analysis.csv"
)

print(
    "6. feature_importance.csv"
)

print(
    "7. risk_distribution.png"
)

print(
    "8. abnormal_labs_by_diagnosis.png"
)

print(
    "9. confusion_matrix.png"
)

print(
    "10. feature_importance.png"
)

print(
    "11. healthcare_risk_model.pkl"
)

print("\n" + "=" * 70)
print("END OF PROGRAM")
print("=" * 70)