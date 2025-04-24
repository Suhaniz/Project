import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Eligibility Predictor")
st.markdown("Enter applicant details below to check loan eligibility.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_amount_term = st.selectbox("Loan Term (in days)", [360, 180, 240, 120])
credit_history = st.selectbox("Credit History", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert categorical inputs to numerical
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1.0 if credit_history == "Yes" else 0.0
dependents = 4 if dependents == "3+" else int(dependents)
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# Make prediction
if st.button("Check Loan Status"):
    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_amount_term, credit_history, property_area]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")