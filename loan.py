import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load(open("rf.pkl", "rb"))

st.title("Loan Approval Prediction")

st.write("Enter applicant details below")

occupation_status_map = {'Employed' : 1 , 'Self-Employed' : 2 , 'Student' : 3}
product_type_map = {'Credit Card' : 1 , 'Personal Loan' : 2 , 'Line of Credit' : 3}
loan_intent_map = {'Personal' : 1, 'Education' : 2, 'Medical' : 3, 'Business' : 4, 'Home Improvement' : 5, 'Debt Consolidation' : 6  }


# Create 3 column layout
col1, col2, col3 = st.columns(3)


with col1:
    age = st.number_input("Age", 18, 100)
    occupation_status = st.selectbox("Occupation Status" , list(occupation_status_map.keys()))
    years_employed = st.number_input("Years Employed", 0, 50)
    annual_income = st.number_input("Annual Income")
    credit_score = st.number_input("Credit Score", 300, 900)
    credit_history_years = st.number_input("Credit History Years")

with col2:
    savings_assets = st.number_input("Savings / Assets")
    current_debt = st.number_input("Current Debt")
    defaults_on_file = st.number_input("Defaults On File", 0, 20)
    delinquencies_last_2yrs = st.number_input("Delinquencies Last 2 Years", 0, 20)
    derogatory_marks = st.number_input("Derogatory Marks", 0, 20)
    product_type = st.selectbox("Product Type" , list(product_type_map.keys()))


with col3:
    loan_intent = st.selectbox("Loan Intent" , list(loan_intent_map.keys()))
    loan_amount = st.number_input("Loan Amount")
    interest_rate = st.number_input("Interest Rate")
    debt_to_income_ratio = st.number_input("Debt to Income Ratio")
    loan_to_income_ratio = st.number_input("Loan to Income Ratio")
    payment_to_income_ratio = st.number_input("Payment to Income Ratio")

# Prediction
if st.button("Predict Loan Status"):

    features = np.array([[age, occupation_status_map[occupation_status], years_employed, annual_income,
                          credit_score, credit_history_years, savings_assets,
                          current_debt, defaults_on_file, delinquencies_last_2yrs,
                          derogatory_marks, product_type_map[product_type], loan_intent_map[loan_intent],
                          loan_amount, interest_rate, debt_to_income_ratio,
                          loan_to_income_ratio, payment_to_income_ratio]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")