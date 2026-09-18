import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).parent

BASE_PATH = Path(__file__).parent

MODEL_PATH = BASE_PATH / "healthcare_risk_model.pkl"
PATIENTS_PATH = BASE_PATH / "outputs" / "processed_patient_data.csv"
FEATURE_IMPORTANCE_PATH = BASE_PATH / "feature_importance.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare Risk Prediction",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD PATIENT DATA
# ============================================================

@st.cache_data
def load_patient_data():

    return pd.read_csv(PATIENTS_PATH)

@st.cache_data
def load_feature_importance():

    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)

    return None

try:

    model = load_model()
    patients = load_patient_data()
    feature_importance = load_feature_importance()

except Exception as e:

    st.error("Unable to load the model or patient data.")

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🏥 Healthcare Risk Prediction Application")

st.markdown(
    """
    ### Interactive Patient Risk Prediction

    This application uses a machine learning model to estimate
    the project-defined risk category of a patient based on
    demographic, diagnostic, treatment and laboratory features.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "🏠 Home",
        "🔮 Risk Prediction",
        "📊 Patient Analysis",
        "📈 Analytics",
        "🤖 Model Information"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("Healthcare Risk Prediction System")

    st.write(
        """
        This application demonstrates an end-to-end healthcare
        data analytics and machine learning workflow.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Patients",
            len(patients)
        )

    with col2:

        st.metric(
            "Total Diagnoses",
            patients["diagnosis_name"].nunique()
        )

    with col3:

        st.metric(
            "Average Age",
            f"{patients['age'].mean():.1f}"
        )

    with col4:

        st.metric(
            "Average Treatment Cost",
            f"₹{patients['treatment_cost'].mean():,.0f}"
        )

    st.divider()

    st.subheader("Project Workflow")

    st.markdown(
        """
        **1. Data Collection**

        Patient, diagnosis, outcome and laboratory datasets.

        **2. Data Cleaning**

        Missing-value checking, datatype conversion and
        duplicate handling.

        **3. Feature Engineering**

        Length of stay, abnormal laboratory counts and
        laboratory summary statistics.

        **4. Machine Learning**

        Random Forest classification model.

        **5. Interactive Prediction**

        Users can select an existing patient or enter
        information for a new patient.

        **6. Analytics**

        Patient and diagnosis-level insights.
        """
    )

    st.info(
        "Note: This is an educational project using synthetic data. "
        "It is not a clinical diagnostic or medical decision-making system."
    )


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "🔮 Risk Prediction":

    st.header("🔮 Interactive Risk Prediction")

    prediction_mode = st.radio(
        "Select prediction mode",
        [
            "Existing Patient",
            "New Patient"
        ],
        horizontal=True
    )

    st.divider()


    # ========================================================
    # EXISTING PATIENT
    # ========================================================

    if prediction_mode == "Existing Patient":

        st.subheader("Select Patient")

        patient_ids = patients["patient_id"].tolist()

        selected_patient_id = st.selectbox(
            "Patient ID",
            patient_ids
        )

        patient = patients[
            patients["patient_id"] == selected_patient_id
        ].iloc[0]


        st.subheader("Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.write("**Patient ID**")
            st.write(patient["patient_id"])

            st.write("**Name**")
            st.write(patient["name"])

            st.write("**Age**")
            st.write(patient["age"])

        with col2:

            st.write("**Gender**")
            st.write(patient["gender"])

            st.write("**Diagnosis**")
            st.write(patient["diagnosis_name"])

            st.write("**Length of Stay**")
            st.write(f"{patient['length_of_stay']} days")

        with col3:

            st.write("**Treatment Cost**")
            st.write(
                f"₹{patient['treatment_cost']:,.2f}"
            )

            st.write("**Abnormal Lab Count**")
            st.write(patient["abnormal_lab_count"])

            st.write("**Total Lab Tests**")
            st.write(patient["total_lab_tests"])


        st.divider()

        st.subheader("Laboratory Summary")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Average Result",
                f"{patient['average_lab_result']:.2f}"
            )

        with col2:

            st.metric(
                "Maximum Result",
                f"{patient['maximum_lab_result']:.2f}"
            )

        with col3:

            st.metric(
                "Minimum Result",
                f"{patient['minimum_lab_result']:.2f}"
            )


        # ====================================================
        # PREDICTION
        # ====================================================

        if st.button(
            "🔮 Predict Patient Risk",
            type="primary",
            use_container_width=True
        ):

            input_data = pd.DataFrame([{

                "age": patient["age"],

                "gender": patient["gender"],

                "diagnosis_name":
                    patient["diagnosis_name"],

                "treatment_cost":
                    patient["treatment_cost"],

                "length_of_stay":
                    patient["length_of_stay"],

                "abnormal_lab_count":
                    patient["abnormal_lab_count"],

                "total_lab_tests":
                    patient["total_lab_tests"],

                "average_lab_result":
                    patient["average_lab_result"],

                "maximum_lab_result":
                    patient["maximum_lab_result"],

                "minimum_lab_result":
                    patient["minimum_lab_result"]

            }])


            prediction = model.predict(input_data)[0]


            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    input_data
                )[0]

                classes = model.classes_

                positive_index = list(classes).index(1)

                risk_probability = (
                    probability[positive_index] * 100
                )

            else:

                risk_probability = None


            st.divider()

            st.subheader("Prediction Result")


            if prediction == 1:

                st.error(
                    "⚠️ HIGHER RISK CATEGORY"
                )

            else:

                st.success(
                    "✅ LOWER RISK CATEGORY"
                )


            if risk_probability is not None:

                st.metric(
                    "Estimated Risk Probability",
                    f"{risk_probability:.2f}%"
                )

                st.progress(
                    min(
                        int(risk_probability),
                        100
                    )
                )
                
            # ====================================================
            # DOWNLOAD PREDICTION REPORT
            # ====================================================

            risk_category = (
                "Higher Risk"
                if prediction == 1
                else "Lower Risk"
            )

            report = pd.DataFrame([{

                "Patient ID": patient["patient_id"],
                "Patient Name": patient["name"],
                "Age": patient["age"],
                "Gender": patient["gender"],
                "Diagnosis": patient["diagnosis_name"],
                "Treatment Cost": patient["treatment_cost"],
                "Length of Stay": patient["length_of_stay"],
                "Abnormal Lab Count": patient["abnormal_lab_count"],
                "Total Lab Tests": patient["total_lab_tests"],
                "Average Lab Result": patient["average_lab_result"],
                "Maximum Lab Result": patient["maximum_lab_result"],
                "Minimum Lab Result": patient["minimum_lab_result"],
                "Predicted Risk": risk_category,
                "Risk Probability": (
                    f"{risk_probability:.2f}%"
                    if risk_probability is not None
                    else "N/A"
                )

            }])

            csv_data = report.to_csv(index=False)

            st.download_button(
                label="📥 Download Prediction Report",
                data=csv_data,
                file_name=(
                    f"patient_{patient['patient_id']}"
                    "_risk_prediction.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )


    # ========================================================
    # NEW PATIENT
    # ========================================================

    else:

        st.subheader("Enter New Patient Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=40
            )

            gender = st.selectbox(
                "Gender",
                sorted(
                    patients["gender"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            diagnosis = st.selectbox(
                "Diagnosis",
                sorted(
                    patients["diagnosis_name"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            treatment_cost = st.number_input(
                "Treatment Cost",
                min_value=0.0,
                value=25000.0,
                step=1000.0
            )

            length_of_stay = st.number_input(
                "Length of Stay (days)",
                min_value=0,
                max_value=365,
                value=5
            )

        with col2:

            abnormal_lab_count = st.number_input(
                "Abnormal Lab Count",
                min_value=0,
                value=1
            )

            total_lab_tests = st.number_input(
                "Total Lab Tests",
                min_value=0,
                value=3
            )

            average_lab_result = st.number_input(
                "Average Lab Result",
                min_value=0.0,
                value=50.0
            )

            maximum_lab_result = st.number_input(
                "Maximum Lab Result",
                min_value=0.0,
                value=80.0
            )

            minimum_lab_result = st.number_input(
                "Minimum Lab Result",
                min_value=0.0,
                value=20.0
            )


        st.divider()


        if st.button(
            "🔮 Predict New Patient Risk",
            type="primary",
            use_container_width=True
        ):

            input_data = pd.DataFrame([{

                "age": age,

                "gender": gender,

                "diagnosis_name": diagnosis,

                "treatment_cost": treatment_cost,

                "length_of_stay": length_of_stay,

                "abnormal_lab_count": abnormal_lab_count,

                "total_lab_tests": total_lab_tests,

                "average_lab_result": average_lab_result,

                "maximum_lab_result": maximum_lab_result,

                "minimum_lab_result": minimum_lab_result

            }])


            prediction = model.predict(
                input_data
            )[0]


            probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                classes = model.classes_

                if 1 in classes:

                    positive_index = list(
                        classes
                    ).index(1)

                    probability = (
                        probabilities[positive_index]
                        * 100
                    )


            st.divider()

            st.subheader("Prediction Result")


            if prediction == 1:

                st.error(
                    "⚠️ HIGHER RISK CATEGORY"
                )

            else:

                st.success(
                    "✅ LOWER RISK CATEGORY"
                )


            if probability is not None:

                st.metric(
                    "Estimated Risk Probability",
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(int(probability), 100)
                )

            st.caption(
                "This prediction is generated by the trained "
                "machine learning model and is intended only "
                "for educational analysis."
            )
            # ====================================================
            # DOWNLOAD NEW PATIENT PREDICTION
            # ====================================================

            risk_category = (
                "Higher Risk"
                if prediction == 1
                else "Lower Risk"
            )

            report = pd.DataFrame([{
                "Patient Type": "New Patient",
                "Age": age,
                "Gender": gender,
                "Diagnosis": diagnosis,
                "Treatment Cost": treatment_cost,
                "Length of Stay": length_of_stay,
                "Abnormal Lab Count": abnormal_lab_count,
                "Total Lab Tests": total_lab_tests,
                "Average Lab Result": average_lab_result,
                "Maximum Lab Result": maximum_lab_result,
                "Minimum Lab Result": minimum_lab_result,
                "Predicted Risk": risk_category,
                "Risk Probability": (
                    f"{probability:.2f}%"
                    if probability is not None
                    else "N/A"
                )
            }])

            csv_data = report.to_csv(index=False)

            st.download_button(
                label="📥 Download Prediction Report",
                data=csv_data,
                file_name="new_patient_risk_prediction.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# PATIENT ANALYSIS
# ============================================================

elif page == "📊 Patient Analysis":

    st.header("📊 Patient Analysis")

    search_id = st.number_input(
        "Enter Patient ID",
        min_value=int(
            patients["patient_id"].min()
        ),
        max_value=int(
            patients["patient_id"].max()
        ),
        value=int(
            patients["patient_id"].min()
        )
    )


    selected = patients[
        patients["patient_id"] == search_id
    ]


    if len(selected) == 0:

        st.warning(
            "Patient ID not found."
        )

    else:

        patient = selected.iloc[0]

        st.subheader(
            f"Patient: {patient['name']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Age",
                patient["age"]
            )

        with col2:

            st.metric(
                "Diagnosis",
                patient["diagnosis_name"]
            )

        with col3:

            st.metric(
                "Treatment Cost",
                f"₹{patient['treatment_cost']:,.0f}"
            )

        with col4:

            st.metric(
                "Length of Stay",
                f"{patient['length_of_stay']} days"
            )


        st.divider()

        st.subheader("Patient Record")

        st.dataframe(
            selected,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.header("📈 Healthcare Analytics")

    st.subheader("Patients by Diagnosis")

    diagnosis_counts = (
        patients["diagnosis_name"]
        .value_counts()
    )

    st.bar_chart(
        diagnosis_counts
    )


    st.subheader("Average Treatment Cost by Diagnosis")

    cost_analysis = (
        patients
        .groupby("diagnosis_name")["treatment_cost"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        cost_analysis
    )


    st.subheader("Risk Category Distribution")

    risk_counts = (
        patients["risk_target"]
        .map({
            0: "Lower Risk",
            1: "Higher Risk"
        })
        .value_counts()
    )

    st.bar_chart(
        risk_counts
    )


    st.subheader("Patient Dataset")

    st.dataframe(
        patients,
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.header("🤖 Machine Learning Model")

    st.write(
        """
        This section provides information about the machine
        learning model used by the application.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------------

    st.subheader("Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

    with col2:

        st.metric(
            "Training Patients",
            "250"
        )

    with col3:

        st.metric(
            "Features",
            "10"
        )

    with col4:

        st.metric(
            "Task",
            "Binary Classification"
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("Features Used by the Model")

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

    feature_df = pd.DataFrame({
        "Feature": features
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    st.subheader("Prediction Target")

    st.write(
        """
        The model performs binary classification using the
        project-defined risk target.
        """
    )

    st.markdown(
        """
        **Lower Risk Category:**  
        Recovered, Improved, Stable

        **Higher Risk Category:**  
        Referred, Deceased
        """
    )

    st.info(
        """
        The risk categories in this project are analytical
        categories created for this synthetic dataset. They
        should not be interpreted as clinical diagnoses or
        medical risk assessments.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("Feature Importance")

    if feature_importance is not None:

        st.dataframe(
            feature_importance,
            use_container_width=True,
            hide_index=True
        )

        if (
            "feature" in feature_importance.columns
            and "importance" in feature_importance.columns
        ):

            chart_data = (
                feature_importance
                .sort_values(
                    "importance",
                    ascending=True
                )
                .set_index("feature")
            )

            st.bar_chart(
                chart_data["importance"]
            )

    else:

        st.warning(
            "Feature importance file was not found."
        )
        
        
    # to run the app, use the following command in your terminal:
    # cd ".\healthcare_sample_data_200plus_csv"
    # python -m streamlit run app.py
    
    
    # cd ".\healthcare_sample_data_200plus_csv"
    # C:\Users\jatin\myenv\Scripts\python.exe -m streamlit --version
    # C:\Users\jatin\myenv\Scripts\python.exe -m streamlit run app.py