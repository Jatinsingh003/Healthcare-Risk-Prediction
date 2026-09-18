# Healthcare Risk Analysis & Interactive Prediction

An end-to-end healthcare data analytics and machine learning project using PostgreSQL, Python, Pandas, Scikit-learn, Joblib, Matplotlib, and Streamlit.

> **Important:** This project uses synthetic/educational data. The risk categories and predictions are not clinical diagnoses, medical advice, or a validated healthcare decision-support system.

## Project Overview

The project combines patient, diagnosis, outcome, and laboratory datasets to demonstrate:

- Data cleaning and preprocessing
- Relational data integration
- Feature engineering
- Exploratory data analysis
- Laboratory abnormality analysis
- Machine learning classification
- Model evaluation
- Patient-level risk prediction
- Interactive Streamlit application
- Downloadable prediction reports

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- PostgreSQL

## Project Structure

```text
Healthcare-Risk-Prediction/
│
├── app.py
├── Risk_model.py
├── Healthcare.sql
├── README.md
├── requirements.txt
│
├── patients.csv
├── diagnosis.csv
├── outcomes.csv
├── labs.csv
│
├── models/
│   └── healthcare_risk_model.pkl
│
└── outputs/
    ├── abnormal_lab_results.csv
    ├── abnormal_labs_by_diagnosis.png
    ├── confusion_matrix.png
    ├── diagnosis_analysis.csv
    ├── feature_importance.csv
    ├── feature_importance.png
    ├── higher_risk_patients.csv
    ├── patient_risk_report.csv
    ├── processed_patient_data.csv
    └── risk_distribution.png
```

Root-level duplicate model/output files may be retained locally as backup copies. The organized copies under `models/` and `outputs/` should be treated as the preferred application paths.

## Workflow

```text
Raw CSV Data
     ↓
Data Cleaning
     ↓
Data Integration
     ↓
Feature Engineering
     ↓
Exploratory Analysis
     ↓
Risk Target Creation
     ↓
Random Forest Model
     ↓
Model Evaluation
     ↓
Saved Model (.pkl)
     ↓
Streamlit Application
     ↓
Interactive Prediction
```

## Data

The project uses four related datasets:

- `patients.csv` — patient demographics, admission/discharge information, diagnosis and outcome IDs, and treatment cost
- `diagnosis.csv` — diagnosis lookup information
- `outcomes.csv` — outcome lookup information
- `labs.csv` — laboratory test results and normal ranges

The raw CSV files should remain unchanged by the analysis pipeline.

## Feature Engineering

The project derives features including:

- Length of hospital stay
- Abnormal laboratory count
- Total laboratory tests
- Average laboratory result
- Maximum laboratory result
- Minimum laboratory result

Laboratory results are compared with the normal ranges supplied by the dataset.

## Risk Target

For this educational project, outcomes are grouped into a project-defined binary target.

**Lower Risk Category**
- Recovered
- Improved
- Stable

**Higher Risk Category**
- Referred
- Deceased

This is a project-specific analytical classification, not a clinical definition of patient risk.

## Machine Learning

The current model is a **Random Forest Classifier**.

The feature set includes:

```text
age
gender
diagnosis_name
treatment_cost
length_of_stay
abnormal_lab_count
total_lab_tests
average_lab_result
maximum_lab_result
minimum_lab_result
```

Categorical variables are handled using `OneHotEncoder` within a Scikit-learn preprocessing pipeline.

The trained pipeline is saved with Joblib and loaded by the Streamlit application.

## Model Evaluation

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Feature Importance

## Streamlit Application

`app.py` provides an interactive interface for:

- Healthcare dashboard
- Existing patient selection
- New patient prediction
- Patient analysis
- Risk probability
- Model information
- Feature importance
- Downloadable prediction reports

## Running the Project

### 1. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the application

```powershell
python -m streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

### 4. Retrain the model

If the model needs to be regenerated:

```powershell
python Risk_model.py
```

This should regenerate the trained model and analytical outputs according to the paths defined in the script.

## Modeling Consideration: Potential Target Leakage

Some current features, particularly **treatment cost** and **length of hospital stay**, may contain information that becomes available during or after treatment. Such variables may therefore not be suitable for an early-admission prediction system.

They are retained here because this is an educational machine learning demonstration using synthetic data.

For a real-world early-risk system, features should be selected according to a clearly defined prediction time point and should contain only information available at that time.

## Limitations

- The dataset is synthetic and relatively small.
- The risk target is project-defined and not clinically validated.
- Performance on synthetic data does not establish real-world clinical performance.
- The current feature set may contain information unavailable at an early prediction point.
- The application must not be used for medical diagnosis, treatment decisions, or clinical risk assessment.

## Data Privacy

Only synthetic or appropriately authorized data should be placed in a public repository.

Never upload real patient medical records or personally identifiable health information to a public GitHub repository.

## Future Improvements

- Redesign features for admission-time prediction
- Use a larger and more representative dataset
- Compare multiple machine learning algorithms
- Apply cross-validation and hyperparameter tuning
- Improve probability calibration
- Automate laboratory input and validation
- Add prediction history
- Improve application UI/UX
- Deploy the Streamlit application

## Author

**Jatin**

Portfolio project demonstrating:

**Data Analysis • Python • SQL • Machine Learning • Data Visualization • Streamlit • Healthcare Analytics**

## License

This project is intended for educational and portfolio purposes. Add an open-source license if you want to define how others may use, modify, and distribute the code.
