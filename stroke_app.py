import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and columns
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# 🎨 Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f5f9;
        background-image: linear-gradient(to bottom right, #f0f5f9, #dbe9f4);
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.title("🧠 Stroke Risk Predictor")
st.write("Enter your health information to assess your risk of stroke.")

# Input widgets
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
ever_married = st.selectbox("Have you ever been married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value
