import streamlit as st
import numpy as np
import joblib

# Load model, scaler, and column names
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.markdown("""
    <h1 style='text-align: center; color: teal;'>🧠 Stroke Risk Predictor</h1>
    <h4 style='text-align: center; color: gray;'>Your personalized health companion</h4>
    <hr>
""", unsafe_allow_html=True)

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])

with col2:
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Glucose & BMI
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)

# Submit
if st.button("🔍 Predict Stroke Risk"):
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

    # Combine with 0s first
    full_input = {col: 0 for col in columns}
    full_input.update(input_dict)
    for key in cat_features:
        if key in full_input:
            full_input[key] = 1

    input_array = np.array([list(full_input.values())])
    input_scaled = scaler.transform(input_array)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    # Classification
    if prob < 0.34:
        risk = "🟢 Low Risk"
        st.success("🟢 You are at low risk. Keep up a healthy lifestyle!")
    elif prob < 0.67:
        risk = "🟡 Moderate Risk"
        st.warning("🟡 You are at moderate risk. Consider lifestyle changes.")
    else:
        risk = "🔴 High Risk"
        st.error("🔴 You are at high risk. Consult a doctor immediately.")

    # Output
    st.markdown(f"""
        <h3>Prediction: {risk}</h3>
        <p style='color: gray;'>(Probability: {prob:.2f})</p>
    """, unsafe_allow_html=True)

    st.progress(int(prob * 100))

    # Recommendations
    st.markdown("""---""")
    st.markdown("### 💡 Personalized Health Recommendations")
    if prob < 0.34:
        st.markdown("""
        - 😁 Maintain your current lifestyle  
        - ✅ Regular check-ups  
        - 💪 Stay active  
        """)
    elif prob < 0.67:
        st.markdown("""
        - 🥗 Improve your diet (less sugar & salt)  
        - 🚶 Increase daily activity  
        - 🩺 Visit a physician for advice  
        """)
    else:
        st.markdown("""
        - 🩺 Book a health consultation immediately  
        - 🚫 Avoid smoking and alcohol  
        - 🧘 Practice stress management  
        """)
