import os
import pathlib
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependencies
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # graceful fallback

# OCR utils are optional; app works without them
try:
    from ocr_utils import (
        extract_text_from_file,
        extract_electricity_bill,
        extract_fuel_type,
        calculate_eco_score,
    )
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -------------------------------
# Page config & simple theming
# -------------------------------
st.set_page_config(
    page_title="EcoCred – Loan & Schemes",
    page_icon="🌱",
    layout="wide",
)

# Minimal CSS polish
st.markdown(
    """
    <style>
      .ekgmqs4 { gap: .5rem; } /* reduce column gaps */
      .metric-card { padding: 1rem; border-radius: 10px; background: #f6f9fc; border: 1px solid #e9eef3; }
      .tag { display:inline-block; margin:2px 6px 2px 0; padding:2px 8px; font-size:12px; border-radius:12px; background:#eef7ee; color:#256029; }
      .small-muted { color:#6b7280; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌱 EcoCred — Dual-Mode Loan Approval & Schemes")

# -------------------------------
# Constants and helpers (aligned with backend semantics)
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


def normalize_for_model(data: dict, region: str) -> dict:
    money_keys = [
        "loan_amount", "annual_income", "monthly_bills", "past_loans_total_principal",
        "past_loans_late_fee", "past_loans_interest", "revol_bal", "last_pymnt_amnt",
    ]
    out = data.copy()
    if region.upper() == "IN":
        for k in money_keys:
            if k in out:
                out[k] = to_float(out[k]) * USD_PER_INR
    else:
        for k in money_keys:
            if k in out:
                out[k] = to_float(out[k])
    return out


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    rate = df["requested_interest"] / 100.0
    term = df["term_months"]

    safe_rate = np.where(rate <= 0, 0.01, rate)
    safe_term = np.where(term <= 0, 36, term)

    df["installment"] = df["loan_amount"] * (safe_rate / 12) / (1 - (1 + safe_rate / 12) ** (-safe_term))
    df["out_prncp"] = df["loan_amount"]
    df["out_prncp_inv"] = df["loan_amount"]

    df["dti"] = (df["monthly_bills"]) / np.maximum(df["annual_income"] / 12.0, 1.0)
    df["revol_util_num"] = df.get("revol_util", 0)
    df["job_type_code"] = df["job_type"].map(JOB_TYPE_MAPPING)

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

    df["sub_grade_code"] = df.get("sub_grade", "A3")
    df["sub_grade_code"] = df["sub_grade_code"].map(SUB_GRADE_MAPPING)

    df["eco_score"] = df.get("eco_score", 0.5)
    fv = df.get("fuel_type", "Unknown")
    if isinstance(fv, (list, tuple, pd.Series)):
        fv = str(fv[0]) if len(fv) else "Unknown"
    fuel_val = str(fv).lower()
    df["is_EV"] = 1 if fuel_val == "electric" else 0

    df["total_acc"] = df.get("total_acc", 5)
    df["open_acc"] = df.get("open_acc", 3)
    df["mths_since_last_delinq"] = df.get("mths_since_last_delinq", 24)
    df["last_pymnt_amnt"] = df.get("last_pymnt_amnt", df["installment"])
    df["last_pymnt_d"] = df.get("last_pymnt_d", "2025-08-01")
    df["last_credit_pull_d"] = df.get("last_credit_pull_d", "2025-08-01")
    df["addr_state"] = df.get("addr_state", "NY")
    df["zip_code"] = df.get("zip_code", "10001")
    df["int_rate_num"] = df["requested_interest"]

    return df


def recommend_schemes(region: str, user: dict) -> List[dict]:
    income = to_float(user.get("annual_income", 0))
    loan_amt = to_float(user.get("loan_amount", 0))
    eco_score = to_float(user.get("eco_score", 0))
    credit_score = to_float(user.get("fico_range_high", 650))
    emp_type = str(user.get("job_type", "salaried")).lower()
    gender = str(user.get("gender", "")).lower()
    dti = to_float(user.get("dti_display", 0))
    emp_years = to_float(user.get("emp_length_years", 0))
    is_ev = int(user.get("is_EV", 0))

    recs: List[dict] = []

    def add(name, desc, link=None, tags=None):
        recs.append({
            "name": name,
            "description": desc,
            "link": link or "",
            "tags": tags or [],
        })

    if region.upper() == "IN":
        if eco_score >= 0.8 and is_ev:
            add("FAME/EV Subsidy", "Central/state incentives for EV purchase; lowers on-road cost.",
                "https://www.fame2.in", ["EV", "Green"]) 
        if eco_score >= 0.7 and loan_amt >= 10e5:
            add("SBI Green Home Loan", "Preferential rates for energy-efficient homes.",
                "https://sbi.co.in", ["Home", "Green"]) 
        if income <= 2e5 and loan_amt <= 1e5:
            add("PM SVANidhi", "Working capital loans for street vendors/small traders.",
                "https://pmsvanidhi.mohua.gov.in", ["Micro", "Working Capital"]) 
        if income <= 3e5 and emp_type in ["self-employed", "other"]:
            add("MUDRA (Shishu/Kishore/Tarun)", "Micro/small business loans up to ₹10 lakh.",
                "https://www.mudra.org.in", ["MSME"]) 
        if emp_type == "salaried" and loan_amt >= 2e5 and emp_years >= 2:
            add("PM Awas Yojana (CLSS)", "Interest subsidy for eligible first-time home buyers.",
                "https://pmaymis.gov.in", ["Housing", "Subsidy"]) 
        if gender == "female" and emp_type in ["self-employed", "other"]:
            add("Mahila Udyam Nidhi", "SIDBI initiative supporting women entrepreneurs.",
                "https://sidbi.in", ["Women", "MSME"]) 
        if credit_score < 600:
            add("Credit Improvement Counselling", "Plan to improve your creditworthiness before applying.",
                "", ["Credit"]) 
        elif 600 <= credit_score <= 750:
            add("Standard Personal Loan", "Regular rate PL for average credit.",
                "", ["Personal"]) 
        elif credit_score > 750 and dti < 0.35:
            add("Prime Low-Interest PL", "Lower rates for excellent credit & manageable DTI.",
                "", ["Personal", "Prime"]) 
        if loan_amt > 5e5 and income < 8e5:
            add("NSIC Subsidy Support", "Support for small manufacturers & service units.",
                "https://nsic.co.in", ["MSME", "Subsidy"]) 
    elif region.upper() == "US":
        if eco_score >= 0.8 and is_ev:
            add("EV Tax Credit (Federal/State)", "Potential tax credits/rebates for EV purchase.",
                "", ["EV", "Green"]) 
        if eco_score >= 0.7 and loan_amt >= 100000:
            add("Green Home / Energy Efficient Mortgage", "Lender programs for energy-efficient homes.",
                "", ["Home", "Green"]) 
        if dti >= 0.4:
            add("Debt Consolidation Loan", "Consolidate high-interest debt to lower DTI.",
                "", ["DTI"]) 
        if income <= 25000:
            add("CDFI (Community)", "Alternative lenders serving low-income borrowers.",
                "", ["Community", "Alt-credit"]) 
        if credit_score < 600:
            add("Credit Builder Loan", "Grow score with small secured installment loans.",
                "", ["Credit"]) 
        elif credit_score > 750 and dti < 0.35:
            add("Prime Personal Loan", "Best rates for strong credit and low DTI.",
                "", ["Personal", "Prime"]) 
    else:
        if eco_score >= 0.8 and is_ev:
            add("NRI EV Loan (Bank)", "EV purchase financing in India for NRIs.", "", ["NRI", "EV"]) 
        if loan_amt >= 5e5 and credit_score >= 700:
            add("NRI Home Loan (Bank)", "Home purchase in India for NRIs.", "", ["NRI", "Home"]) 
        if emp_type in ["self-employed", "other"]:
            add("NRI Business/Startup Loan", "For setting up/expanding business in India.", "", ["NRI", "Business"]) 

    return recs


@st.cache_resource(show_spinner=False)
def load_model() -> Optional[object]:
    if joblib is None:
        return None
    candidates = [
        pathlib.Path(__file__).parent / "models" / "lending_club_model_1.pkl",
        pathlib.Path.cwd() / "backend" / "models" / "lending_club_model_1.pkl",
        pathlib.Path.cwd() / "models" / "lending_club_model_1.pkl",
    ]
    for p in candidates:
        if p.exists():
            try:
                return joblib.load(str(p))
            except Exception:
                continue
    return None


# -------------------------------
# Sidebar - Region & About
# -------------------------------
with st.sidebar:
    st.header("Settings")
    region = st.radio("Choose Region", options=["IN", "US"], index=0, horizontal=True)
    currency = "₹" if region == "IN" else "$"

    st.caption(
        "Model runs in USD internally. In India mode, amounts convert to USD for prediction;"
        " scheme recommendations are localized."
    )

    st.divider()
    st.markdown("**About EcoCred**")
    st.markdown(
        "A demo that blends traditional credit inputs with eco footprint to suggest approvals and schemes."
    )

# -------------------------------
# OCR Section (optional)
# -------------------------------
st.subheader("1) Upload Docs (optional)")
eco_score: Optional[float] = None
electricity_units: Optional[int] = None
fuel_type: Optional[str] = None

col1, col2 = st.columns(2)

with col1:
    elec_file = st.file_uploader(
        "Electricity Bill (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"], key="bill"
    )
    if elec_file:
        if not OCR_AVAILABLE:
            st.info("OCR libraries not installed. Skipping OCR.")
        else:
            try:
                tmp_path = pathlib.Path("uploads") / elec_file.name
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(elec_file.getvalue())
                text = extract_text_from_file(str(tmp_path))
                units = extract_electricity_bill(str(tmp_path))
                electricity_units = int(units) if units else 0
                st.success(f"Extracted Units: {electricity_units}")
                with st.expander("Bill OCR Preview"):
                    st.text(text[:2000])
            except Exception as e:
                st.error(f"OCR (bill) error: {e}")

with col2:
    rc_file = st.file_uploader(
        "Vehicle RC (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"], key="rc"
    )
    if rc_file:
        if not OCR_AVAILABLE:
            st.info("OCR libraries not installed. Skipping OCR.")
        else:
            try:
                tmp_path = pathlib.Path("uploads") / rc_file.name
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(rc_file.getvalue())
                text = extract_text_from_file(str(tmp_path))
                fuel_type = extract_fuel_type(str(tmp_path))
                eco_score = calculate_eco_score(fuel_type)
                st.success(
                    f"Fuel Type: {fuel_type or 'Unknown'} | Eco Score: {eco_score if eco_score is not None else 'N/A'}"
                )
                with st.expander("RC OCR Preview"):
                    st.text(text[:2000])
            except Exception as e:
                st.error(f"OCR (RC) error: {e}")

# -------------------------------
# Loan Input Form
# -------------------------------
st.subheader("2) Enter Details")

with st.form("loan_form"):
    colA, colB = st.columns(2)
    with colA:
        loan_amount = st.number_input(
            f"Loan Amount ({currency})",
            min_value=1000.0 if region == "US" else 10000.0,
            value=20000.0 if region == "US" else 200000.0,
        )
        term_months = st.selectbox("Loan Term (months)", [36, 60], index=0)
        requested_interest = st.number_input(
            "Requested Interest (%)", min_value=1.0, max_value=25.0, value=12.5
        )

    with colB:
        annual_income = st.number_input(
            f"Annual Income ({currency})",
            min_value=1000.0 if region == "US" else 50000.0,
            value=40000.0 if region == "US" else 600000.0,
        )
        monthly_bills = st.number_input(
            f"Monthly Bills ({currency})", min_value=0.0,
            value=float(electricity_units) if electricity_units else (1200.0 if region == "US" else 8000.0),
        )

    colC, colD = st.columns(2)
    with colC:
        emp_length_years = st.number_input("Years in Job", min_value=0, max_value=50, value=5, step=1)
        job_type = st.selectbox("Job Type", ["salaried", "self-employed", "other"], index=0)
        gender = st.selectbox("Gender (optional)", ["", "male", "female", "other"], index=0)
    with colD:
        past_loans_total_principal = st.number_input(
            f"Past Loans Principal ({currency})", min_value=0.0,
            value=10000.0 if region == "US" else 150000.0,
        )
        past_loans_late_fee = st.number_input(
            f"Past Loans Late Fee ({currency})", min_value=0.0,
            value=50.0 if region == "US" else 500.0,
        )
        past_loans_interest = st.number_input(
            f"Past Loans Interest ({currency})", min_value=0.0,
            value=500.0 if region == "US" else 7000.0,
        )

    st.text_input(
        "Eco Score (auto from RC)", value=str(eco_score) if eco_score is not None else "Not extracted", disabled=True
    )

    zip_code = st.text_input("ZIP/Postal Code", value="10001" if region == "US" else "560001")
    addr_state = st.text_input("State", value="NY" if region == "US" else "KA")

    submitted = st.form_submit_button("Check Approval")

# -------------------------------
# Prediction (local model or fallback)
# -------------------------------
if submitted:
    payload = {
        "region": region,
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
        "addr_state": addr_state,
    }

    # Build DTI for display (region units)
    ai_disp = to_float(payload.get("annual_income", 0))
    mb_disp = to_float(payload.get("monthly_bills", 0))
    dti_display = float(mb_disp / max(ai_disp / 12.0, 1.0)) if ai_disp else 0.0

    # Model normalize -> features
    normalized = normalize_for_model(payload.copy(), region)
    df = pd.DataFrame([normalized])
    df = calculate_features(df)

    # Ensure model feature set (robust to missing cols)
    features = [
        "total_rec_prncp", "last_pymnt_d", "last_pymnt_amnt", "loan_amount", "installment",
        "out_prncp", "last_fico_range_high", "total_rec_late_fee", "total_rec_int",
        "credit_history_months", "total_pymnt", "dti", "last_credit_pull_d", "zip_code",
        "revol_util_num", "annual_income", "total_pymnt_inv", "recoveries", "revol_bal",
        "mths_since_last_delinq", "int_rate_num", "out_prncp_inv", "open_acc", "addr_state",
        "fico_range_low", "total_acc", "emp_length_years", "sub_grade_code", "term_months", "last_fico_range_low",
    ]
    for f in features:
        if f not in df.columns:
            df[f] = 0
    X = df[features].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    model = load_model()

    def fallback_probability(payload_: dict) -> float:
        # Simple, deterministic heuristic as a placeholder when model is missing
        income = to_float(payload_.get("annual_income", 0))
        loan = to_float(payload_.get("loan_amount", 0))
        dti = dti_display
        eco = to_float(payload_.get("eco_score", 0.5))
        is_ev = 1.0 if str(payload_.get("fuel_type", "")).lower() == "electric" else 0.0
        base = 0.5 + 0.2 * eco + 0.1 * is_ev
        affordability = np.tanh((income / 12.0) / max(loan / max(payload_.get("term_months", 36), 1), 1))
        penalty = 0.4 * np.tanh(max(dti - 0.3, 0) * 2)
        prob = base * 0.6 + affordability * 0.5 - penalty
        return float(np.clip(prob, 0.01, 0.99))

    if model is not None:
        try:
            # Some models may expose predict_proba; handle gracefully
            if hasattr(model, "predict_proba"):
                prob_default = float(model.predict_proba(X)[:, 1][0])
                approval_prob = float(1.0 - prob_default)
            else:
                # Fallback to decision_function/predict
                y_score = None
                if hasattr(model, "decision_function"):
                    y_score = float(model.decision_function(X)[0])
                elif hasattr(model, "predict"):
                    y_score = float(model.predict(X)[0])
                approval_prob = float(1 / (1 + np.exp(-y_score))) if y_score is not None else fallback_probability(payload)
        except Exception as e:
            st.warning(f"Model prediction failed, using heuristic: {e}")
            approval_prob = fallback_probability(payload)
    else:
        st.info("Model file not found. Using a heuristic scorer.")
        approval_prob = fallback_probability(payload)

    approval_class = int(approval_prob >= 0.5)

    # Enrich for schemes
    enriched = payload.copy()
    enriched["dti_display"] = dti_display
    enriched["is_EV"] = int(str(payload.get("fuel_type", "")).lower() == "electric")

    schemes = recommend_schemes(region, enriched)

    # -------------------------------
    # UI: Results
    # -------------------------------
    st.subheader("Prediction Result")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Approval Probability", f"{approval_prob * 100:,.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Decision", "Approved" if approval_class == 1 else "Rejected",
                  delta="✅" if approval_class == 1 else "❌")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("DTI (display)", f"{dti_display:.2f}")
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
                st.markdown(f"**{s.get('name','(Unnamed)')}**")
                if s.get("description"):
                    st.caption(s["description"]) 
                if s.get("link"):
                    st.write(s["link"]) 
                tags = s.get("tags", [])
                if tags:
                    st.markdown(" ".join([f"<span class='tag'>{t}</span>" for t in tags]), unsafe_allow_html=True)
                st.markdown("<span class='small-muted'>—</span>", unsafe_allow_html=True)
                st.divider()
