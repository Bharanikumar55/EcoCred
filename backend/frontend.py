import streamlit as st
import requests

st.set_page_config(page_title="EcoCred Loan Prediction", layout="centered")
st.title("EcoCred Loan Approval Predictor")

st.write("Enter your details below to check loan approval:")

# -------------------- User Inputs --------------------
with st.form("loan_form"):
    # Loan info
    loan_amount = st.number_input("Loan Amount", min_value=1000, value=20000)
    term_months = st.selectbox("Loan Term (months)", [36, 60])
    requested_interest = st.number_input("Requested Interest (%)", min_value=1.0, max_value=25.0, value=12.5)

    # Income & debt
    annual_income = st.number_input("Annual Income", min_value=1000, value=40000)
    monthly_bills = st.number_input("Monthly Bills", min_value=0, value=1200)

    # Employment
    emp_length_years = st.number_input("Years in Job", min_value=0, max_value=50, value=5)
    job_type = st.selectbox("Job Type", ["salaried", "self-employed", "other"])

    # Past loans
    past_loans_total_principal = st.number_input("Past Loans Principal", min_value=0, value=10000)
    past_loans_late_fee = st.number_input("Past Loans Late Fee", min_value=0, value=50)
    past_loans_interest = st.number_input("Past Loans Interest", min_value=0, value=500)

    # Eco / green
    eco_score = st.slider("Eco Score (0–1)", 0.0, 1.0, 0.85)
    ev_ownership = st.radio("EV Ownership", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    # Location
    zip_code = st.text_input("ZIP Code", value="10001")
    addr_state = st.text_input("State", value="NY")

    # Submit button
    submitted = st.form_submit_button("Check Approval")

# -------------------- Send POST request to API --------------------
if submitted:
    input_data = {
        "loan_amount": loan_amount,
        "term_months": term_months,
        "requested_interest": requested_interest,
        "annual_income": annual_income,
        "monthly_bills": monthly_bills,
        "emp_length_years": emp_length_years,
        "job_type": job_type,
        "past_loans_total_principal": past_loans_total_principal,
        "past_loans_late_fee": past_loans_late_fee,
        "past_loans_interest": past_loans_interest,
        "eco_score": eco_score,
        "ev_ownership": ev_ownership,
        "zip_code": zip_code,
        "addr_state": addr_state
    }

    try:
        url = "http://127.0.0.1:5000/predict"
        response = requests.post(url, json=input_data)
        result = response.json()

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Approval Probability:** {result['approval_probability']*100:.2f}%")
        st.write(f"**Approval Class:** {'Approved' if result['approval_class']==1 else 'Rejected'}")
        st.write(f"**Recommendations:** {', '.join(result['recommendations']) if result['recommendations'] else 'None'}")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
