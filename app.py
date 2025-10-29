import streamlit as st
import pandas as pd
import joblib

st.title("Diabetes Prediction App 🩺")
st.write("Enter the patient data and select a model to predict the risk of diabetes.")

# --- Model selection ---
model_choice = st.selectbox("Choose model", ["LightGBM", "XGBoost"])

# --- Load model based on selection ---
if model_choice == "LightGBM":
    model = joblib.load("lightgbm_pipeline.pkl")
else:
    model = joblib.load("xgboost_pipeline.pkl")

# --- User inputs ---
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
smoking_history = st.selectbox("Smoking History", ["current", "ever", "former", "never", "not current"])

# --- Input DataFrame ---
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level],
    'gender': [gender],
    'smoking_history': [smoking_history]
})

# --- Prediction ---
if st.button("Predict Diabetes"):
    prediction = model.predict(input_data)[0]
    
    probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        msg = f"⚠️ HIGH risk of diabetes."
        if probability is not None:
            msg += f" (Probability: {probability:.2f})"
        st.error(msg)
    else:
        msg = f"✅ LOW risk of diabetes."
        if probability is not None:
            msg += f" (Probability: {probability:.2f})"
        st.success(msg)
