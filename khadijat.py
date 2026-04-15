import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------
# -----------------------------
@st.cache_resource  
def load_model_and_scaler():
    # Ensure these files are in the SAME folder as this script
    model = tf.keras.models.load_model('burnout_model.keras')
    scaler = joblib.load('scaler.pkl')
    
    model_columns = [
        'Designation', 
        'Resource Allocation', 
        'Mental Fatigue Score',
        'Gender_Male', 
        'Company Type_Service', 
        'WFH Setup Available_Yes'
    ]
    return model, scaler, model_columns

model, scaler, model_columns = load_model_and_scaler()

# -----------------------------
# Tittle
# -----------------------------
st.set_page_config(page_title="Burnout Analysis", page_icon="🔥")
st.title("🔥 Burnout Presentation: Model 1")
st.markdown("#Numerical Data Analysis & Prediction")
# input
col1, col2 = st.columns(2)
with col1:
    designation = st.slider("Designation Level (1-5)", 1, 5, 2)
    resource = st.slider("Resource Allocation (1-10)", 1.0, 10.0, 5.0)
    fatigue = st.slider("Mental Fatigue Score (0-10)", 0.0, 10.0, 5.0)
with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    company = st.selectbox("Company Type", ["Product", "Service"])
    wfh = st.selectbox("WFH Available", ["No", "Yes"])

# -----------------------------
# -----------------------------
if st.button("Calculate Burnout Risk"):
    try:
        input_data = pd.DataFrame({
            'Designation': [designation],
            'Resource Allocation': [resource],
            'Mental Fatigue Score': [fatigue],
            'Gender_Male': [1 if gender == "Male" else 0],
            'Company Type_Service': [1 if company == "Service" else 0],
            'WFH Setup Available_Yes': [1 if wfh == "Yes" else 0]
        })

        input_encoded = input_data[model_columns]

    
        if list(input_encoded.columns) != list(scaler.feature_names_in_):
            st.warning("⚠️ Column Mismatch Detected!")
            st.write("Your Scaler wants:", list(scaler.feature_names_in_))
            st.write("Your App sent:", list(input_encoded.columns))
            st.stop() 
        # ------------------------------

        scaled_input = scaler.transform(input_encoded)
        prediction = model.predict(scaled_input, verbose=0)
        
        #  result is between 0 and 1
        burnout_score = np.clip(float(prediction[0][0]), 0.0, 1.0)

    
        st.divider()
        st.metric(label="Predicted Burnout Score", value=f"{burnout_score:.2f}")
        
        if burnout_score > 0.7:
            st.error("**High Risk:** Burnout likely. Immediate intervention recommended.")
        elif burnout_score > 0.4:
            st.warning("**Moderate Risk:** Employee is showing signs of fatigue.")
        else:
            st.success("**Low Risk:** Balanced workload detected.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check that your model_columns list matches your training features exactly.")

# -----------------------------
#  Presentation Sidebar
# -----------------------------
st.sidebar.header("Presentation Master Plan")
st.sidebar.write("✅ Model 1: Burnout (ANN)")
st.sidebar.write("⬜ Model 2: Digits (CNN)")
st.sidebar.write("⬜ Model 3: Emotion (NLP)")