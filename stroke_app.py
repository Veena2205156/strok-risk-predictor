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
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.title("🧠 Stroke Risk Predictor")
st.write("Enter your health details below to check your risk level for stroke:")

# 📋 User Inputs
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

# 🔘 Predict Button
if st.button("Predict Stroke Risk"):

    # Prepare input
    input_dict = {
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
    }

    # One-hot encoding for categorical inputs
    cat_features = {
        f"gender_{gender}": 1,
        f"ever_married_{ever_married}": 1,
        f"work_type_{work_type}": 1,
        f"Residence_type_{residence_type}": 1,
        f"smoking_status_{smoking_status}": 1
    }

    # Combine all inputs into final format
    full_input = {col: 0 for col in columns}
    full_input.update(input_dict)
    for key in cat_features:
        if key in full_input:
            full_input[key] = 1

    # Convert to array and scale
    input_array = np.array([list(full_input.values())])
    input_scaled = scaler.transform(input_array)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    # Classification logic
    def classify_risk(prob):
        if prob < 0.34:
            return "🟢 Low Risk"
        elif prob < 0.67:
            return "🟡 Moderate Risk"
        else:
            return "🔴 High Risk"

    risk = classify_risk(prob)

    # Display results
    st.subheader(f"Prediction: {risk}")
    st.caption(f"Model confidence: {prob:.2f}")

    # 🚦 Progress bar
    st.progress(prob)

    # 🩺 Health Tips
    st.markdown("### 🩺 Personalized Health Tips")
    if prob < 0.34:
        st.success("✅ You're at low risk! Keep maintaining a healthy lifestyle. 💪")
        st.markdown("""
        - Continue regular exercise  
        - Maintain a balanced diet  
        - Get regular check-ups  
        - Avoid smoking and excess alcohol  
        """)
    elif prob < 0.67:
        st.warning("⚠️ Moderate risk. Consider some lifestyle changes.")
        st.markdown("""
        - Reduce salt and sugar intake  
        - Start moderate daily exercise  
        - Monitor your blood pressure  
        - Consult your doctor for regular assessments  
        """)
    else:
        st.error("🚨 High risk detected! Please take medical advice seriously.")
        st.markdown("""
        - Visit a healthcare provider ASAP  
        - Manage diabetes, hypertension, or heart disease actively  
        - Quit smoking and limit alcohol  
        - Maintain a healthy weight and stress levels  
        """)

