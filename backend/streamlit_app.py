# streamlit_app.py
import os
import pathlib
from typing import Optional, List, Dict, Any
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import joblib
import requests

# Optional: SHAP (graceful fallback)
try:
    import shap
except Exception:
    shap = None

# Load .env (API keys / model names)
load_dotenv()

# -------------------------------
# Page config & theme (green)
# -------------------------------
st.set_page_config(page_title="EcoCred — Loan & Schemes", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
      .ekgmqs4 { gap: .5rem; }
      .metric-card { padding: 1rem; border-radius: 10px; background: #f6fff2; border: 1px solid #e6ffea; }
      .tag { display:inline-block; margin:2px 6px 2px 0; padding:4px 10px; font-size:12px; border-radius:12px; background:#e6ffea; color:#0b6b2e; }
      .small-muted { color:#6b7280; font-size: 12px; }
      .stApp { background: linear-gradient(180deg, #ffffff 0%, #f3fff6 100%); }
      .title { font-weight:700; color: #0b6b2e; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌱 EcoCred — Dual-Mode Loan Approval & Schemes")

# -------------------------------
# Helpers, constants
# -------------------------------
USD_PER_INR = 1 / 83.0
JOB_TYPE_MAPPING = {"salaried": 0, "self-employed": 1, "other": 2}
SUB_GRADE_MAPPING = {
    "A1": 0, "A2": 1, "A3": 2, "A4": 3, "A5": 4,
    "B1": 5, "B2": 6, "B3": 7, "B4": 8, "B5": 9,
    "C1": 10, "C2": 11, "C3": 12, "C4": 13, "C5": 14,
}

def to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def safe_load_joblib(path: pathlib.Path):
    try:
        return joblib.load(str(path))
    except Exception as e:
        st.warning(f"Failed to load model at {path}: {e}")
        return None

# -------------------------------
# Model loaders (US and India)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model_us() -> Optional[Any]:
    # Candidate locations for the US model
    candidates = [
        pathlib.Path(_file_).parent / "models" / "lending_club_model_1.pkl",
        pathlib.Path.cwd() / "backend" / "models" / "lending_club_model_1.pkl",
        pathlib.Path.cwd() / "models" / "lending_club_model_1.pkl",
    ]
    for p in candidates:
        if p.exists():
            return safe_load_joblib(p)
    return None

@st.cache_resource(show_spinner=False)
def load_model_in() -> Optional[Any]:
    # Candidate locations for the India model (user provided path)
    candidates = [
        pathlib.Path(_file_).parent / "models" / "EcoCred_model.pkl",
        pathlib.Path.cwd() / "backend" / "models" / "EcoCred_model.pkl",
        pathlib.Path.cwd() / "models" / "EcoCred_model.pkl",
        pathlib.Path("C:/Users/jites/Desktop/EcoCred/backend/models/EcoCred_model.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return safe_load_joblib(p)
    return None

# -------------------------------
# Sidebar: region selector + info
# -------------------------------
with st.sidebar:
    st.header("Settings")
    region = st.radio("Choose Region / Model", options=["IN (EcoCred)", "US (LendingClub)"], index=0)
    use_region = "IN" if region.startswith("IN") else "US"
    st.caption("Pick India to use EcoCred_model, or US to use Lending Club model.")
    st.divider()
    st.markdown("*About*")
    st.markdown("EcoCred demo — blends standard credit inputs with eco footprint for approval & scheme suggestions.")
    st.divider()
    st.markdown("*Chatbot Settings*")
    st.caption("Put keys in .env: GEMINI_API_KEY, GEMINI_MODEL, GEMINI_API_URL (optional).")

# -------------------------------
# OCR utils (optional)
# -------------------------------
try:
    from ocr_utils import extract_text_from_file, extract_electricity_bill, extract_fuel_type, calculate_eco_score
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -------------------------------
# Input forms
# -------------------------------
st.subheader("1) Upload Docs (optional)")
eco_score = None
electricity_units = None
fuel_type = None

c1, c2 = st.columns(2)
with c1:
    elec_file = st.file_uploader("Electricity Bill (jpg/png/pdf)", type=["jpg","jpeg","png","pdf"])
    if elec_file:
        if not OCR_AVAILABLE:
            st.info("OCR not installed — skip/uploading will still work.")
        else:
            tmp = pathlib.Path("uploads") / elec_file.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(elec_file.getvalue())
            try:
                text = extract_text_from_file(str(tmp))
                units = extract_electricity_bill(str(tmp))
                electricity_units = int(units) if units else None
                st.success(f"Extracted electricity units: {electricity_units}")
            except Exception as e:
                st.error(f"OCR error: {e}")

with c2:
    rc_file = st.file_uploader("Vehicle RC (jpg/png/pdf)", type=["jpg","jpeg","png","pdf"])
    if rc_file:
        if not OCR_AVAILABLE:
            st.info("OCR not installed — skipping RC extraction.")
        else:
            tmp = pathlib.Path("uploads") / rc_file.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(rc_file.getvalue())
            try:
                text = extract_text_from_file(str(tmp))
                fuel_type = extract_fuel_type(str(tmp))
                eco_score = calculate_eco_score(fuel_type)
                st.success(f"Fuel Type: {fuel_type or 'Unknown'} | Eco Score: {eco_score if eco_score is not None else 'N/A'}")
            except Exception as e:
                st.error(f"OCR (RC) error: {e}")

# -------------------------------
# Two separate forms depending on region
# -------------------------------
st.subheader("2) Enter Details")
submitted = False

if use_region == "US":
    # US form (original)
    with st.form("us_form"):
        colA, colB = st.columns(2)
        with colA:
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, value=20000.0)
            term_months = st.selectbox("Loan Term (months)", [36, 60], index=0)
            requested_interest = st.number_input("Requested Interest (%)", min_value=1.0, max_value=25.0, value=12.5)
        with colB:
            annual_income = st.number_input("Annual Income ($)", min_value=1000.0, value=40000.0)
            monthly_bills = st.number_input("Monthly Bills ($)", min_value=0.0, value=float(electricity_units) if electricity_units else 1200.0)
        colC, colD = st.columns(2)
        with colC:
            emp_length_years = st.number_input("Years in Job", min_value=0, max_value=50, value=5, step=1)
            job_type = st.selectbox("Job Type", ["salaried","self-employed","other"], index=0)
            gender = st.selectbox("Gender (optional)", ["","male","female","other"], index=0)
        with colD:
            past_loans_total_principal = st.number_input("Past Loans Principal ($)", min_value=0.0, value=10000.0)
            past_loans_late_fee = st.number_input("Past Loans Late Fee ($)", min_value=0.0, value=50.0)
            past_loans_interest = st.number_input("Past Loans Interest ($)", min_value=0.0, value=500.0)
        st.text_input("Eco Score (auto from RC)", value=str(eco_score) if eco_score is not None else "Not extracted", disabled=True)
        zip_code = st.text_input("ZIP/Postal Code", value="10001")
        addr_state = st.text_input("State", value="NY")
        submitted = st.form_submit_button("Check Approval")
else:
    # India form (fields tuned to Indian model)
    with st.form("in_form"):
        colA, colB = st.columns(2)
        with colA:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=10000.0, value=200000.0)
            term_months = st.selectbox("Loan Term (months)", [36, 60], index=0)
            requested_interest = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5)
        with colB:
            annual_income = st.number_input("Annual Income (₹)", min_value=50000.0, value=600000.0)
            monthly_bills = st.number_input("Monthly Bills (₹)", min_value=0.0, value=float(electricity_units) if electricity_units else 8000.0)
        colC, colD = st.columns(2)
        with colC:
            emp_length = st.selectbox("Employment length", ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"], index=5)
            emp_title = st.text_input("Employer / Title", value="")
            application_type = st.selectbox("Application Type", ["INDIVIDUAL","JOINT"], index=0)
        with colD:
            grade = st.selectbox("Grade (A/B/C)", ["A","B","C"], index=0)
            home_ownership = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE","OTHER"], index=0)
            last_payment_date = st.text_input("Last Payment Date (dd-mm-yyyy)", value="")
        st.text_input("Eco Score (auto from RC)", value=str(eco_score) if eco_score is not None else "Not extracted", disabled=True)
        zip_code = st.text_input("ZIP/Postal Code", value="560001")
        addr_state = st.text_input("State", value="KA")
        submitted = st.form_submit_button("Check Approval (India model)")

# -------------------------------
# Utility: build feature vector for India (EcoCred_model)
# -------------------------------
def prepare_india_features(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a DataFrame that matches the features your India model expects.
    You told me the important features include:
      ['total_payment', 'installment', 'loan_amount', 'int_rate_num', 'annual_income', 'dti',
       'total_acc', 'term_num', 'loan_to_income_ratio', 'installment_to_income_ratio',
       'grade_x_int_rate', 'term_x_loan_to_income', 'installment_x_dti']
    We compute as many of these as possible from the payload.
    """
    loan_amount = to_float(payload.get("loan_amount", 0))
    annual_income = to_float(payload.get("annual_income", 1))
    term_months = to_float(payload.get("term_months", 36))
    requested_interest = to_float(payload.get("requested_interest", 0.0))
    monthly_bills = to_float(payload.get("monthly_bills", 0.0))

    int_rate_num = requested_interest / 100.0
    term_num = term_months
    installment = 0.0
    try:
        r = int_rate_num
        n = max(term_num, 1)
        monthly_rate = r / 12.0
        installment = loan_amount * (monthly_rate) / (1 - (1 + monthly_rate) ** (-n)) if r > 0 else loan_amount / n
    except Exception:
        installment = loan_amount / max(term_num, 1)

    loan_to_income_ratio = loan_amount / max(annual_income, 1.0)
    installment_to_income_ratio = installment / max(annual_income / 12.0, 1.0)
    dti = monthly_bills / max(annual_income / 12.0, 1.0)

    # Map grade to numeric factor (simple mapping)
    grade_str = str(payload.get("grade", "A")).upper()
    grade_factor = {"A": 0, "B": 1, "C": 2}.get(grade_str, 0)
    grade_x_int_rate = grade_factor * int_rate_num
    term_x_loan_to_income = term_num * loan_to_income_ratio
    installment_x_dti = installment * dti

    # Some placeholders for features we don't have: total_payment, total_acc
    total_payment = 0.0  # unknown without payment history
    total_acc = 5.0

    feats = {
        "total_payment": total_payment,
        "installment": installment,
        "loan_amount": loan_amount,
        "int_rate_num": int_rate_num,
        "annual_income": annual_income,
        "dti": dti,
        "total_acc": total_acc,
        "term_num": term_num,
        "loan_to_income_ratio": loan_to_income_ratio,
        "installment_to_income_ratio": installment_to_income_ratio,
        "grade_x_int_rate": grade_x_int_rate,
        "term_x_loan_to_income": term_x_loan_to_income,
        "installment_x_dti": installment_x_dti,
    }
    df = pd.DataFrame([feats])
    return df

# -------------------------------
# Utility: build feature vector for US (Lending Club)
# -------------------------------
def prepare_us_features(payload: Dict[str, Any]) -> pd.DataFrame:
    # Normalize money to USD if region set to IN earlier; but for US keep as-is.
    normalized = payload.copy()
    df = pd.DataFrame([normalized])
    # Reuse original calculate_features logic (subset) here:
    # some features are computed in the earlier code; replicate minimal set
    rate = df["requested_interest"] / 100.0
    term = df["term_months"]
    safe_rate = np.where(rate <= 0, 0.01, rate)
    safe_term = np.where(term <= 0, 36, term)
    df["installment"] = df["loan_amount"] * (safe_rate / 12) / (1 - (1 + safe_rate / 12) ** (-safe_term))
    df["out_prncp"] = df["loan_amount"]
    df["out_prncp_inv"] = df["loan_amount"]
    df["dti"] = df["monthly_bills"] / np.maximum(df["annual_income"] / 12.0, 1.0)
    df["revol_util_num"] = df.get("revol_util", 0)
    try:
        df["job_type_code"] = df["job_type"].map(JOB_TYPE_MAPPING)
    except Exception:
        df["job_type_code"] = 0
    df["total_rec_prncp"] = df.get("past_loans_total_principal", 0)
    df["total_rec_late_fee"] = df.get("past_loans_late_fee", 0)
    df["total_rec_int"] = df.get("past_loans_interest", 0)
    df["credit_history_months"] = df.get("credit_history_months", df["emp_length_years"] * 12)
    df["total_pymnt"] = df["total_rec_prncp"] + df["total_rec_int"] + df["total_rec_late_fee"]
    df["total_pymnt_inv"] = df["total_pymnt"]
    df["recoveries"] = df.get("recoveries", 0)
    df["fico_range_low"] = df.get("fico_range_low", 600)
    df["fico_range_high"] = df.get("fico_range_high", 700)
    df["last_fico_range_low"] = df.get("last_fico_range_low", df["fico_range_low"])
    df["last_fico_range_high"] = df.get("last_fico_range_high", df["fico_range_high"])
    df["sub_grade_code"] = df.get("sub_grade", pd.Series(["A3"] * len(df))).map(SUB_GRADE_MAPPING)
    df["eco_score"] = df.get("eco_score", 0.5)
    fv = df.get("fuel_type", "Unknown")
    if isinstance(fv, (list, tuple, pd.Series)):
        fv = str(fv[0]) if len(fv) else "Unknown"
    df["is_EV"] = 1 if str(fv).lower() == "electric" else 0
    df["total_acc"] = df.get("total_acc", 5)
    df["open_acc"] = df.get("open_acc", 3)
    df["mths_since_last_delinq"] = df.get("mths_since_last_delinq", 24)
    df["last_pymnt_amnt"] = df.get("last_pymnt_amnt", df["installment"])
    df["last_pymnt_d"] = df.get("last_pymnt_d", "2025-08-01")
    df["last_credit_pull_d"] = df.get("last_credit_pull_d", "2025-08-01")
    df["addr_state"] = df.get("addr_state", "NY")
    df["zip_code"] = df.get("zip_code", "10001")
    df["int_rate_num"] = df["requested_interest"]
    # Pick the features used by the US model (we use the larger list you provided earlier)
    features = [
        "total_rec_prncp","last_pymnt_d","last_pymnt_amnt","loan_amount","installment",
        "out_prncp","last_fico_range_high","total_rec_late_fee","total_rec_int",
        "credit_history_months","total_pymnt","dti","last_credit_pull_d","zip_code",
        "revol_util_num","annual_income","total_pymnt_inv","recoveries","revol_bal",
        "mths_since_last_delinq","int_rate_num","out_prncp_inv","open_acc","addr_state",
        "fico_range_low","total_acc","emp_length_years","sub_grade_code","term_months","last_fico_range_low"
    ]
    for f in features:
        if f not in df.columns:
            df[f] = 0
    X = df[features].copy()
    # ensure numeric columns are numeric
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X

# -------------------------------
# Prediction + SHAP logic
# -------------------------------
shap_summary_text = ""
# -------------------------------
# Fallback probability (heuristic)
# -------------------------------

def fallback_probability(payload: dict, dti: float) -> float:
    """
    Simple heuristic probability if model is missing or fails.
    Uses DTI and eco_score to compute a score between 0 and 1.
    """
    eco_score = float(payload.get("eco_score", 0.5))
    # lower DTI is better
    dti_factor = max(0.0, min(1.0, 1.0 - dti))  
    # eco score factor (higher is better)
    eco_factor = max(0.0, min(1.0, eco_score))
    
    # Combine with weights (tweakable)
    prob = 0.6 * dti_factor + 0.4 * eco_factor
    # Clip to 0-1
    prob = max(0.0, min(1.0, prob))
    return prob

if submitted:
    # prepare payload depending on region
    if use_region == "IN":
        payload = {
            "loan_amount": loan_amount,
            "term_months": term_months,
            "requested_interest": requested_interest,
            "annual_income": annual_income,
            "monthly_bills": monthly_bills,
            "emp_length": emp_length,
            "emp_title": emp_title,
            "application_type": application_type,
            "grade": grade,
            "home_ownership": home_ownership,
            "last_payment_date": last_payment_date,
            "eco_score": float(eco_score) if eco_score is not None else 0.5,
            "zip_code": zip_code,
            "addr_state": addr_state
        }
        X_in = prepare_india_features(payload)
        model_in = load_model_in()
        if model_in is None:
            st.info("India model not found — using heuristic.")
            approval_prob = fallback_probability(payload, X_in["dti"].iloc[0])
        else:
            try:
                if hasattr(model_in, "predict_proba"):
                    p = model_in.predict_proba(X_in)[:, 1][0]
                    approval_prob = float(p)  # Assuming model trained for positive label probability
                else:
                    y_score = float(model_in.predict(X_in)[0])
                    approval_prob = float(1/(1+np.exp(-y_score)))
            except Exception as e:
                st.warning(f"India model prediction failed: {e}")
                approval_prob = fallback_probability(payload, X_in["dti"].iloc[0])

        approval_class = int(approval_prob >= 0.5)

        # SHAP for India model (attempt)
        shap_summary_text = ""
        if shap is not None and model_in is not None:
            try:
                # Try TreeExplainer for tree-based models; fall back to generic Explainer
                try:
                    explainer = shap.TreeExplainer(model_in, feature_perturbation="tree_path_dependent")
                except Exception:
                    explainer = shap.Explainer(model_in, X_in)
                shap_values = explainer(X_in)
                st.subheader("Feature Impact (SHAP) — India model")
                shap.initjs()
                fig = plt.figure(figsize=(8, 4))
                shap.summary_plot(shap_values, X_in, show=False)
                plt.tight_layout()
                plt.savefig("shap_india.png", bbox_inches="tight")
                st.image("shap_india.png")
                plt.close(fig)

                shap_importances = pd.DataFrame({
                    "feature": X_in.columns,
                    "impact": np.abs(shap_values.values).mean(axis=0)
                }).sort_values(by="impact", ascending=False)
                top_features = shap_importances.head(5)
                shap_summary_text = "Top features impacting approval:\n"
                for _, r in top_features.iterrows():
                    shap_summary_text += f"- {r['feature']}: impact {r['impact']:.4f}\n"
            except Exception as e:
                st.warning(f"SHAP explanation (India) failed: {e}")

    else:
        # US flow
        payload = {
            "region": use_region,
            "loan_amount": loan_amount,
            "term_months": term_months,
            "requested_interest": requested_interest,
            "annual_income": annual_income,
            "monthly_bills": monthly_bills,
            "emp_length_years": emp_length_years,
            "job_type": job_type,
            "gender": gender,
            "past_loans_total_principal": past_loans_total_principal,
            "past_loans_late_fee": past_loans_late_fee,
            "past_loans_interest": past_loans_interest,
            "eco_score": float(eco_score) if eco_score is not None else 0.5,
            "fuel_type": fuel_type if fuel_type else "Unknown",
            "zip_code": zip_code,
            "addr_state": addr_state
        }
        X_us = prepare_us_features(payload)
        model_us = load_model_us()
        if model_us is None:
            st.info("US model not found — using heuristic.")
            approval_prob = fallback_probability(payload, X_us["dti"].iloc[0])
        else:
            try:
                if hasattr(model_us, "predict_proba"):
                    prob_default = float(model_us.predict_proba(X_us)[:, 1][0])
                    approval_prob = float(1.0 - prob_default)  # your earlier code used 1 - prob_default
                else:
                    y_score = float(model_us.predict(X_us)[0])
                    approval_prob = float(1 / (1 + np.exp(-y_score)))
            except Exception as e:
                st.warning(f"US model prediction failed: {e}")
                approval_prob = fallback_probability(payload, X_us["dti"].iloc[0])

        approval_class = int(approval_prob >= 0.5)

        # SHAP for US model
        shap_summary_text = ""
        if shap is not None and model_us is not None:
            try:
                try:
                    explainer = shap.TreeExplainer(model_us, feature_perturbation="tree_path_dependent")
                except Exception:
                    explainer = shap.Explainer(model_us, X_us)
                shap_values = explainer(X_us)
                st.subheader("Feature Impact (SHAP) — US model")
                shap.initjs()
                fig = plt.figure(figsize=(8, 4))
                shap.summary_plot(shap_values, X_us, show=False)
                plt.tight_layout()
                plt.savefig("shap_us.png", bbox_inches="tight")
                st.image("shap_us.png")
                plt.close(fig)

                shap_importances = pd.DataFrame({
                    "feature": X_us.columns,
                    "impact": np.abs(shap_values.values).mean(axis=0)
                }).sort_values(by="impact", ascending=False)
                top_features = shap_importances.head(5)
                shap_summary_text = "Top features impacting approval:\n"
                for _, r in top_features.iterrows():
                    shap_summary_text += f"- {r['feature']}: impact {r['impact']:.4f}\n"
            except Exception as e:
                st.warning(f"SHAP explanation (US) failed: {e}")

    # -------------------------------
    # Recommended schemes (simple)
    # -------------------------------
    def recommend_schemes_simple(region_code: str, enriched: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs = []
        eco = to_float(enriched.get("eco_score", 0.5))
        loan_amt = to_float(enriched.get("loan_amount", 0))
        dti = to_float(enriched.get("dti_display", 0))
        if region_code == "IN":
            if eco >= 0.8:
                recs.append({"name":"FAME/EV Subsidy", "description":"EV incentives (Central/state).", "tags":["EV","Green"]})
            if loan_amt >= 1_000_000:
                recs.append({"name":"SBI Green Home Loan", "description":"Preferential rates for energy-efficient homes.", "tags":["Home","Green"]})
        else:
            if eco >= 0.8:
                recs.append({"name":"EV Tax Credit", "description":"Federal/state EV incentives", "tags":["EV","Green"]})
            if dti >= 0.4:
                recs.append({"name":"Debt Consolidation Loan", "description":"Helps lower DTI", "tags":["Debt","DTI"]})
        return recs

    enriched = payload.copy()
    enriched["dti_display"] = (monthly_bills / max(annual_income / 12.0, 1.0)) if annual_income else 0.0
    enriched["is_EV"] = 1 if str(fuel_type or "").lower() == "electric" else 0
    schemes = recommend_schemes_simple("IN" if use_region == "IN" else "US", enriched)

    # -------------------------------
    # Show metrics & schemes
    # -------------------------------
    st.subheader("Prediction Result")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Approval Probability", f"{approval_prob * 100:,.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Decision", "Approved" if approval_class == 1 else "Rejected", delta="✅" if approval_class == 1 else "❌")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("DTI (display)", f"{enriched.get('dti_display', 0):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Eco Score", f"{to_float(payload.get('eco_score', 0.5)):.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Recommended Schemes")
    if not schemes:
        st.write("No matching schemes right now.")
    else:
        for s in schemes:
            with st.container():
                st.markdown(f"{s.get('name','(Unnamed)')}")
                if s.get("description"):
                    st.caption(s["description"])
                tags = s.get("tags", [])
                if tags:
                    st.markdown(" ".join([f"<span class='tag'>{t}</span>" for t in tags]), unsafe_allow_html=True)
                st.divider()


# -------------------------------
# Hybrid Chatbot Section
# -------------------------------
st.subheader("💬 Ask the EcoCred AI Chatbot (Hybrid)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_input = st.text_input("Type your question here...", key="chat_input")

def shap_knowledge_reply(shap_text: str, schemes: list) -> str:
    """Generate actionable advice from SHAP features and recommended schemes."""
    advice_lines = []

    # SHAP-based advice
    if shap_text:
        for line in shap_text.split("\n"):
            if line.startswith("-"):
                feat = line.split(":")[0].strip("- ").replace("_", " ").title()
                if "Dti" in feat or "Loan To Income" in feat or "Installment To Income" in feat:
                    advice_lines.append(f"✅ Try to reduce {feat.lower()}.")
                else:
                    advice_lines.append(f"✅ Try to increase {feat.lower()}.")

    # Recommended schemes
    if schemes:
        advice_lines.append("\n💡 Recommended schemes:")
        for s in schemes:
            advice_lines.append(f"- {s.get('name')} ({', '.join(s.get('tags',[]))})")

    if not advice_lines:
        return "Focus on stable income, low DTI, and a good eco score for better approval chances."

    return "\n".join(advice_lines)


if st.button("Send", key="chat_send") and chat_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": chat_input})
    reply_parts = []

    # Optional: SHAP summary injection
    shap_summary_text_local = globals().get("shap_summary_text", "")
    try:
        # Load Gemini API key & model
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"

        if gemini_key:
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": gemini_key  # Use API key
            }

            payload = {
                "contents": [{"parts": [{"text": chat_input}]}]
            }

            resp = requests.post(gemini_url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            resp_json = resp.json()

            # Extract Gemini reply
            gemini_reply = None
            if "candidates" in resp_json and len(resp_json["candidates"]) > 0:
                gemini_reply = resp_json["candidates"][0]["content"]["parts"][0]["text"]

            if gemini_reply:
                reply_parts.append(gemini_reply)
            else:
                # fallback to SHAP
                fallback_text = shap_knowledge_reply(shap_summary_text_local, globals().get("schemes", []))
                reply_parts.append(fallback_text)
        else:
            fallback_text = shap_knowledge_reply(shap_summary_text_local, globals().get("schemes", []))
            reply_parts.append(fallback_text)

    except Exception as e:
        # Gemini failed, fallback to SHAP + schemes
        fallback_text = shap_knowledge_reply(shap_summary_text_local, globals().get("schemes", []))
        reply_parts.append(f"Chat API failed: {e}\n\n{fallback_text}")

    final_reply = "\n\n".join(reply_parts).strip()
    if not final_reply:
        final_reply = "I'm here to help — ask about your approval, eco-score, or recommendations."

    st.session_state.chat_history.append({"role": "bot", "content": final_reply})

# Display chat history
for h in st.session_state.chat_history:
    if h["role"] == "user":
         st.markdown(
                f"<div style='background:#eaffea;padding:10px;border-radius:10px;margin-bottom:5px;'><b>👤 You:</b><br>{h['content']}</div>",
                unsafe_allow_html=True,
            )
    else:
       st.markdown(
                f"<div style='background:#f6fff9;padding:10px;border-radius:10px;margin-bottom:10px;'><b>🤖 EcoCred Bot:</b><br>{h['content']}</div>",
                unsafe_allow_html=True,
            )

st.info("Tip: This chatbot tries Gemini 2.5 first. If it fails, it falls back to knowledge-based advice using SHAP + schemes.")