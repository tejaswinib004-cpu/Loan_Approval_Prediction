import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. LOAD ASSETS ---
# Ensure these files are in the same directory as app.py
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- 2. DEFINE MAPPINGS ---
# These MUST match the label encoding used during training
mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'Married': {'No': 0, 'Yes': 1},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education': {'Graduate': 0, 'Not Graduate': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
}

# --- 3. UI SETUP ---
st.set_page_config(page_title="Loan Predictor", page_icon="💰")
st.title("💰 Loan Approval Prediction App")
st.write("Fill in the details below to check if your loan will be approved.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gen = st.selectbox("Gender", ["Male", "Female"])
    mar = st.selectbox("Married", ["Yes", "No"])
    dep = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    edu = st.selectbox("Education", ["Graduate", "Not Graduate"])
    sem = st.selectbox("Self Employed", ["No", "Yes"])
    pro = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    api = st.number_input("Applicant Income", min_value=0, value=0)
    cpi = st.number_input("Coapplicant Income", min_value=0, value=0)
    lam = st.number_input("Loan Amount", min_value=0, value=0)
    lat = st.number_input("Loan Term (Days)", min_value=0, value=0)
    crh = st.selectbox("Credit History", [1.0, 0.0])

# --- 4. PREDICTION LOGIC ---
if st.button("Check Approval Status", use_container_width=True):
    # Feature Engineering (must match training)
    total_income = api + cpi
    
    # Create feature list in the EXACT order the model expects
    features = [
        mappings['Gender'][gen],
        mappings['Married'][mar],
        mappings['Dependents'][dep],
        mappings['Education'][edu],
        mappings['Self_Employed'][sem],
        api,
        cpi,
        lam,
        lat,
        crh,
        mappings['Property_Area'][pro],
        total_income
    ]
    
    # Process and Predict
    input_array = np.array(features).reshape(1, -1)
    scaled_data = scaler.transform(input_array)
    prediction = model.predict(scaled_data)
    
    # --- 5. DISPLAY RESULTS ---
    st.divider()
    if prediction[0] == 1:
        st.success("### ✅ Loan Approved!")
        st.balloons()
    else:
        st.error("### ❌Loan Rejected")