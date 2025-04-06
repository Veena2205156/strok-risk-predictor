import streamlit as st
import numpy as np
import joblib

# 🌈 Custom CSS Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to right, #e0f7fa, #f0f8ff);
        background-size: cover;
        padding: 2rem;
    }

    h1, h3, h4, h2 {
        color: #004d40;
    }

    .stButton > button {
        background-color: #ff4d4d;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #e60000;
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📦 Load model and preprocessing tools
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# 🧠 Title
st.title("🧠 Stroke Risk Predictor")
st.write("Enter the following health details to check your stroke risk level:")

# 📋 Input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
ever_married = st.selectbox("Have you ever been married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked",_
