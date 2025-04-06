import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and columns
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# 🎨 Custom Background Styling
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
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st
