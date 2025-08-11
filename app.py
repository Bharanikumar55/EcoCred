# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from utils import preprocess_input, get_model_and_explainer, shap_reason_engine, generate_chatbot_message, recommend_schemes

st.set_page_config(page_title="EcoCred Demo", layout="centered")

@st.cache_resource
def load_all():
    model, explainer, threshold, feature_cols = get_model_and_explainer()
    return model, explainer, threshold, feature_cols

model, explainer, threshold, feature_cols = load_all()

st.title("EcoCred — Loan Eligibility & Explainability")
st.markdown("Upload a CSV with user rows or input a single user manually.")

# ------------------------
# Manual input
# ------------------------
st.sidebar.header("Manual input")
income = st.sidebar.number_input("Income", value=60000.0)
loan_amount = st.sidebar.number_input("Loan amount", value=15000.0)
monthly_units = st.sidebar.number_input("Monthly units (electricity)", value=300)
vehicle_type = st.sidebar.selectbox("Vehicle type", options=["bike", "car"])
vehicle_map = {"bike": 0, "car": 1}
fuel_type = st.sidebar.selectbox("Fuel type", ["Electric", "Petrol", "Diesel"])
eco_score = st.sidebar.number_input("Eco score (0-20)", min_value=0, max_value=20, value=10)
credit_score = st.sidebar.number_input("Credit score", min_value=300, max_value=900, value=700)
job_type = st.sidebar.selectbox("Job Type", ["Govt", "MNC", "Self"])
loan_history = st.sidebar.selectbox("Loan History", ["Paid", "Ongoing", "No History"])

if st.sidebar.button("Run model for manual input"):
    raw = {
        "income": income,
        "loan_amount": loan_amount,
        "monthly_units": monthly_units,
        "vehicle_type": vehicle_map[vehicle_type],
        "fuel_type": fuel_type,
        "eco_score": eco_score,
        "credit_score": credit_score,
        "job_type": job_type,
        "loan_history": loan_history
    }

    X_user = preprocess_input(raw)
    proba = model.predict_proba(X_user)[:, 1][0]
    pred = int(proba >= threshold)

    st.write(f"Predicted: **{'Approved' if pred==1 else 'Rejected'}** (prob={proba:.3f})")

    # ✅ SHAP reasoning matches prediction
    reason = shap_reason_engine(X_user, explainer, class_index=pred)
    st.subheader("Why this decision?")
    st.write(generate_chatbot_message(reason))
    st.subheader("SHAP table (feature, value, shap_value)")
    st.dataframe(pd.DataFrame(reason['harmful'] + reason['helpful']))

    # Scheme recommendations
    schemes = recommend_schemes(X_user.iloc[0].to_dict())
    st.subheader("💡 Recommended Schemes")
    if schemes:
        for sch in schemes:
            st.markdown(f"**{sch['name']}**\n\n{sch['description']}\n[More info]({sch['link']})\n")
    else:
        st.write("No specific schemes match your profile.")

# ------------------------
# CSV upload
# ------------------------
uploaded = st.file_uploader("Upload CSV (rows with original raw columns)", type=["csv"])
if uploaded is not None:
    df_uploaded = pd.read_csv(uploaded)
    st.write("Preview uploaded rows:")
    st.dataframe(df_uploaded.head())

    if st.button("Run model on uploaded file (first row)"):
        raw_row = df_uploaded.iloc[0].to_dict()
        X_user = preprocess_input(raw_row)
        X_user = pd.DataFrame([X_user.iloc[0]]) if isinstance(X_user, pd.DataFrame) else pd.DataFrame([X_user])

        proba = model.predict_proba(X_user)[:, 1][0]
        pred = int(proba >= threshold)

        st.write(f"Predicted: **{'Approved' if pred==1 else 'Rejected'}** (prob={proba:.3f})")
        
        # ✅ SHAP reasoning matches prediction
        reason = shap_reason_engine(X_user, explainer, class_index=pred)
        st.subheader("Why this decision?")
        st.write(generate_chatbot_message(reason))
        st.dataframe(pd.DataFrame(reason['harmful'] + reason['helpful']))

        # Scheme recommendations
        schemes = recommend_schemes(X_user.iloc[0].to_dict())
        st.subheader("💡 Recommended Schemes")
        if schemes:
            for sch in schemes:
                st.markdown(f"**{sch['name']}**\n\n{sch['description']}\n[More info]({sch['link']})\n")
        else:
            st.write("No specific schemes match your profile.")
