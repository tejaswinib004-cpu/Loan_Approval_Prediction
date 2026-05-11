import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Loan Approval Prediction")

# INPUT FIELDS

Gender = st.selectbox("Gender", ["Male", "Female"])

Married = st.selectbox("Married", ["Yes", "No"])

Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

ApplicantIncome = st.number_input("Applicant Income")

CoapplicantIncome = st.number_input("Coapplicant Income")

LoanAmount = st.number_input("Loan Amount")

Loan_Amount_Term = st.number_input("Loan Amount Term")

Credit_History = st.selectbox("Credit History", [1.0, 0.0])

Property_Area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# =========================================
# FEATURE ENGINEERING
# =========================================

TotalIncome = ApplicantIncome + CoapplicantIncome

# =========================================
# ENCODING
# =========================================

Gender = 1 if Gender == "Male" else 0

Married = 1 if Married == "Yes" else 0

Dependents = {
    "0":0,
    "1":1,
    "2":2,
    "3+":3
}[Dependents]

Education = 0 if Education == "Graduate" else 1

Self_Employed = 1 if Self_Employed == "Yes" else 0

Property_Area = {
    "Rural":0,
    "Semiurban":1,
    "Urban":2
}[Property_Area]

# =========================================
# INPUT ARRAY
# SAME ORDER AS TRAINING DATA
# =========================================

input_data = np.array([[
    
    Gender,
    
    Married,
    
    Dependents,
    
    Education,
    
    Self_Employed,
    
    ApplicantIncome,
    
    CoapplicantIncome,
    
    LoanAmount,
    
    Loan_Amount_Term,
    
    Credit_History,
    
    Property_Area,
    
    TotalIncome
    
]])

# =========================================
# FEATURE SCALING
# =========================================

input_data = scaler.transform(input_data)

# =========================================
# PREDICTION
# =========================================

if st.button("Predict"):
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        
        st.success("Loan Approved")
    
    else:
        
        st.error("Loan Not Approved")