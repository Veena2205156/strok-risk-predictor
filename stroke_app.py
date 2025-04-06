import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and column names
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("🧠 Stroke Risk Predictor")
st.write("Enter the following health details to check your stroke risk level:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
ever_married = st.selectbox("Have you ever been married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Submit
if st.button("Predict Stroke Risk"):
    # Prepare input
    input_dict = {
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
    }

    # One-hot encode
    cat_features = {
        f"gender_{gender}": 1,
        f"ever_married_{ever_married}": 1,
        f"work_type_{work_type}": 1,
        f"Residence_type_{residence_type}": 1,
        f"smoking_status_{smoking_status}": 1
    }

    # Combine all inputs
    full_input = {col: 0 for col in columns}
    full_input.update(input_dict)
    for key in cat_features:
        if key in full_input:
            full_input[key] = 1

    # Convert to numpy array
    input_array = np.array([list(full_input.values())])
    input_scaled = scaler.transform(input_array)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    # Classify risk
    def classify_risk(prob):
        if prob < 0.34:
            return "🟢 Low Risk"
        elif prob < 0.67:
            return "🟡 Moderate Risk"
        else:
            return "🔴 High Risk"

    risk = classify_risk(prob)

    st.subheader(f"Prediction: {risk}")
    st.caption(f"(Probability: {prob:.2f})")
