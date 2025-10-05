from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

# OCR utils
from ocr_utils import extract_electricity_bill, extract_fuel_type, calculate_eco_score, extract_text_from_file

app = Flask(__name__)

# -------------------------------
# Config
# -------------------------------
USD_PER_INR = 1 / 83.0  # ~INR -> USD conversion factor used for model inputs when region=IN
DEFAULT_REGION = "US"   # "US" or "IN"

# -------------------------------
# Load ML model
# -------------------------------
model_path = r'C:\Users\jites\Desktop\EcoCred\backend\models\lending_club_model_1.pkl'
print("Does the model exist?", os.path.exists(model_path))
model = joblib.load(model_path)

# -------------------------------
# Mapping for categorical variables
# -------------------------------
job_type_mapping = {
    'salaried': 0,
    'self-employed': 1,
    'other': 2
}

sub_grade_mapping = {
    'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5': 4,
    'B1': 5, 'B2': 6, 'B3': 7, 'B4': 8, 'B5': 9,
    'C1': 10, 'C2': 11, 'C3': 12, 'C4': 13, 'C5': 14
}

# -------------------------------
# Helpers
# -------------------------------
def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def normalize_for_model(data: dict, region: str) -> dict:
    """
    Keep your model in USD. If region=IN, convert INR -> USD for numeric money fields.
    Non-money fields pass through unchanged.
    """
    money_keys = [
        "loan_amount", "annual_income", "monthly_bills", "past_loans_total_principal",
        "past_loans_late_fee", "past_loans_interest", "revol_bal", "last_pymnt_amnt"
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
    rate = df['requested_interest'] / 100.0
    term = df['term_months']

    # Guard against zero or weird rate/term
    safe_rate = np.where(rate <= 0, 0.01, rate)
    safe_term = np.where(term <= 0, 36, term)

    # installment (safeguarded)
    df['installment'] = df['loan_amount'] * (safe_rate/12) / (1 - (1 + safe_rate/12)**(-safe_term))
    df['out_prncp'] = df['loan_amount']
    df['out_prncp_inv'] = df['loan_amount']

    # Debt to income ratio
    df['dti'] = (df['monthly_bills']) / np.maximum(df['annual_income'] / 12.0, 1.0)

    df['revol_util_num'] = df.get('revol_util', 0)
    df['job_type_code'] = df['job_type'].map(job_type_mapping)

    # Past loan history
    df['total_rec_prncp'] = df.get('past_loans_total_principal', 0)
    df['total_rec_late_fee'] = df.get('past_loans_late_fee', 0)
    df['total_rec_int'] = df.get('past_loans_interest', 0)
    df['credit_history_months'] = df.get('credit_history_months', df['emp_length_years']*12)
    df['total_pymnt'] = df['total_rec_prncp'] + df['total_rec_int'] + df['total_rec_late_fee']
    df['total_pymnt_inv'] = df['total_pymnt']
    df['recoveries'] = df.get('recoveries', 0)

    # Credit info
    df['fico_range_low'] = df.get('fico_range_low', 600)
    df['fico_range_high'] = df.get('fico_range_high', 700)
    df['last_fico_range_low'] = df.get('last_fico_range_low', df['fico_range_low'])
    df['last_fico_range_high'] = df.get('last_fico_range_high', df['fico_range_high'])

    df['sub_grade_code'] = df.get('sub_grade', 'A3')
    df['sub_grade_code'] = df['sub_grade_code'].map(sub_grade_mapping)

    # Environmental factor
    df['eco_score'] = df.get('eco_score', 0.5)
    fv = df.get('fuel_type', 'Unknown')
    if isinstance(fv, (list, tuple, pd.Series)):
        fv = str(fv[0]) if len(fv) else "Unknown"
    fuel_val = str(fv).lower()
    df['is_EV'] = 1 if fuel_val == 'electric' else 0

    # Other defaults
    df['total_acc'] = df.get('total_acc', 5)
    df['open_acc'] = df.get('open_acc', 3)
    df['mths_since_last_delinq'] = df.get('mths_since_last_delinq', 24)
    df['last_pymnt_amnt'] = df.get('last_pymnt_amnt', df['installment'])
    df['last_pymnt_d'] = df.get('last_pymnt_d', '2025-08-01')
    df['last_credit_pull_d'] = df.get('last_credit_pull_d', '2025-08-01')
    df['addr_state'] = df.get('addr_state', 'NY')
    df['zip_code'] = df.get('zip_code', '10001')
    df['int_rate_num'] = df['requested_interest']

    return df

def recommend_schemes(region: str, user: dict) -> list:
    """
    Region-aware scheme engine (IN / US / NRI).
    Uses multi-factor checks: income, loan_amt, dti, credit_score, eco_score, is_ev, emp_years, job_type, gender.
    Note: All numeric inputs here are in the USER'S REGION UNITS (we’ll pass original, not USD-normalized).
    """
    income = to_float(user.get('annual_income', 0))
    loan_amt = to_float(user.get('loan_amount', 0))
    eco_score = to_float(user.get('eco_score', 0))
    credit_score = to_float(user.get('fico_range_high', 650))
    emp_type = str(user.get('job_type', 'salaried')).lower()
    gender = str(user.get('gender', '')).lower()
    dti = to_float(user.get('dti_display', 0))  # dti in user region context (just a ratio)
    emp_years = to_float(user.get('emp_length_years', 0))
    is_ev = int(user.get('is_EV', 0))

    recs = []

    def add(name, desc, link=None, tags=None):
        recs.append({
            "name": name,
            "description": desc,
            "link": link or "",
            "tags": tags or []
        })

    if region.upper() == "IN":
        # Eco-oriented
        if eco_score >= 0.8 and is_ev:
            add("FAME/EV Subsidy",
                "Central/state incentives for EV purchase; lowers on-road cost.",
                "https://www.fame2.in", ["EV", "Green"])
        if eco_score >= 0.7 and loan_amt >= 10e5:
            add("SBI Green Home Loan",
                "Preferential rates for energy-efficient homes.",
                "https://sbi.co.in", ["Home", "Green"])

        # Income & micro loans
        if income <= 2e5 and loan_amt <= 1e5:
            add("PM SVANidhi",
                "Working capital loans for street vendors/small traders.",
                "https://pmsvanidhi.mohua.gov.in", ["Micro", "Working Capital"])
        if income <= 3e5 and emp_type in ['self-employed', 'other']:
            add("MUDRA (Shishu/Kishore/Tarun)",
                "Micro/small business loans up to ₹10 lakh.",
                "https://www.mudra.org.in", ["MSME"])

        # Housing
        if emp_type == 'salaried' and loan_amt >= 2e5 and emp_years >= 2:
            add("PM Awas Yojana (CLSS)",
                "Interest subsidy for eligible first-time home buyers.",
                "https://pmaymis.gov.in", ["Housing", "Subsidy"])

        # Women/Entrepreneurship
        if gender == 'female' and emp_type in ['self-employed', 'other']:
            add("Mahila Udyam Nidhi",
                "SIDBI initiative supporting women entrepreneurs.",
                "https://sidbi.in", ["Women", "MSME"])

        # Credit score oriented
        if credit_score < 600:
            add("Credit Improvement Counselling",
                "Plan to improve your creditworthiness before applying.",
                "", ["Credit"])
        elif 600 <= credit_score <= 750:
            add("Standard Personal Loan",
                "Regular rate PL for average credit.",
                "", ["Personal"])
        elif credit_score > 750 and dti < 0.35:
            add("Prime Low-Interest PL",
                "Lower rates for excellent credit & manageable DTI.",
                "", ["Personal", "Prime"])

        # MSME & subsidy
        if loan_amt > 5e5 and income < 8e5:
            add("NSIC Subsidy Support",
                "Support for small manufacturers & service units.",
                "https://nsic.co.in", ["MSME", "Subsidy"])

    elif region.upper() == "US":
        # Eco-oriented
        if eco_score >= 0.8 and is_ev:
            add("EV Tax Credit (Federal/State)",
                "Potential tax credits/rebates for EV purchase.",
                "", ["EV", "Green"])
        if eco_score >= 0.7 and loan_amt >= 100000:
            add("Green Home / Energy Efficient Mortgage",
                "Lender programs for energy-efficient homes.",
                "", ["Home", "Green"])

        # Income & consolidation
        if dti >= 0.4:
            add("Debt Consolidation Loan",
                "Consolidate high-interest debt to lower DTI.",
                "", ["DTI"])
        if income <= 25000:
            add("Community Development Financial Institution (CDFI)",
                "Alternative lenders serving low-income borrowers.",
                "", ["Community", "Alt-credit"])

        # Credit-based
        if credit_score < 600:
            add("Credit Builder Loan",
                "Grow score with small secured installment loans.",
                "", ["Credit"])
        elif credit_score > 750 and dti < 0.35:
            add("Prime Personal Loan",
                "Best rates for strong credit and low DTI.",
                "", ["Personal", "Prime"])

    else:
        # NRI / Cross-border concept (India-facing borrower abroad)
        if eco_score >= 0.8 and is_ev:
            add("NRI EV Loan (Bank)",
                "EV purchase financing in India for NRIs.",
                "", ["NRI", "EV"])
        if loan_amt >= 5e5 and credit_score >= 700:
            add("NRI Home Loan (Bank)",
                "Home purchase in India for NRIs.",
                "", ["NRI", "Home"])
        if emp_type in ['self-employed', 'other']:
            add("NRI Business/Startup Loan",
                "For setting up/expanding business in India.",
                "", ["NRI", "Business"])

    return recs

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw = request.json or {}
        region = str(raw.get("region", DEFAULT_REGION)).upper()

        # Keep original (display values for region); also create model-ready (USD) copy
        original = raw.copy()
        normalized = normalize_for_model(raw.copy(), region)

        df = pd.DataFrame([normalized])

        # Derived features (model space)
        df = calculate_features(df)

        # Features needed for model
        features = [
            'total_rec_prncp', 'last_pymnt_d', 'last_pymnt_amnt', 'loan_amount', 'installment',
            'out_prncp', 'last_fico_range_high', 'total_rec_late_fee', 'total_rec_int',
            'credit_history_months', 'total_pymnt', 'dti', 'last_credit_pull_d', 'zip_code',
            'revol_util_num', 'annual_income', 'total_pymnt_inv', 'recoveries', 'revol_bal',
            'mths_since_last_delinq', 'int_rate_num', 'out_prncp_inv', 'open_acc', 'addr_state',
            'fico_range_low', 'total_acc', 'emp_length_years', 'sub_grade_code', 'term_months', 'last_fico_range_low'
        ]

        for f in features:
            if f not in df.columns:
                df[f] = 0

        X = df[features].copy()

        # Convert any stray objects to numeric
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        # Predict probability
        prob_default = model.predict_proba(X)[:, 1][0]
        approval_prob = float(1.0 - prob_default)
        approval_class = int(approval_prob >= 0.5)

        # Build a display DTI from the original (region) values
        ai_disp = to_float(original.get('annual_income', 0))
        mb_disp = to_float(original.get('monthly_bills', 0))
        dti_display = float(mb_disp / max(ai_disp / 12.0, 1.0)) if ai_disp else 0.0

        # Enrich original for schemes
        original_enriched = original.copy()
        original_enriched['dti_display'] = dti_display
        original_enriched['is_EV'] = int(str(original.get('fuel_type', '')).lower() == 'electric')

        # Schemes by region
        schemes = recommend_schemes(region, original_enriched)

        return jsonify({
            'region': region,
            'fx_applied': True if region == "IN" else False,
            'fx_rate_INR_per_USD': 1 / USD_PER_INR,
            'approval_probability': round(approval_prob, 4),
            'approval_class': approval_class,
            'dti_display': round(dti_display, 3),
            'schemes': schemes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------
# OCR Routes
# -------------------------------
@app.route('/upload-electricity-bill', methods=['POST'])
def upload_electricity_bill():
    try:
        file = request.files['file']
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        text = extract_text_from_file(file_path)
        units = extract_electricity_bill(file_path)
        return jsonify({'electricity_units': units, 'ocr_preview': text[:2000]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-fuel-doc', methods=['POST'])
def upload_fuel_doc():
    try:
        file = request.files['file']
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        text = extract_text_from_file(file_path)
        fuel_type = extract_fuel_type(file_path)
        eco_score = calculate_eco_score(fuel_type)

        # Optional: incorporate units if sent via form-data for combined score
        units = request.form.get("units")
        if units:
            try:
                units = float(units)
                # naive tweak: lower units -> slightly better eco score
                eco_score = min(1.0, eco_score + (0.05 if units < 150 else 0.0))
            except:
                pass

        return jsonify({'fuel_type': fuel_type, 'eco_score': eco_score, 'ocr_preview': text[:2000]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------
# Run app
# -------------------------------
if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    # use_reloader=False avoids Windows signal error
    app.run(debug=True, use_reloader=False)
