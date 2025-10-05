import streamlit as st
import requests

st.set_page_config(page_title="EcoCred Loan Prediction", layout="centered")
st.title("EcoCred — Dual-Mode Loan Approval & Schemes")

# -------------------- Mode --------------------
region = st.radio("Choose Region", options=["IN", "US"], index=0, horizontal=True)
currency = "₹" if region == "IN" else "$"

st.caption(
    "Model runs in USD internally. In India mode, amounts are converted to USD for prediction; "
    "scheme recommendations are localized."
)

# -------------------- File Uploads for OCR --------------------
st.subheader("1) Upload Docs (optional)")
eco_score = None
electricity_units = None
fuel_type = None

col1, col2 = st.columns(2)

with col1:
    elec_file = st.file_uploader("Electricity Bill (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"], key="bill")
    if elec_file:
        try:
            files = {"file": (elec_file.name, elec_file.getvalue(), elec_file.type or "application/octet-stream")}
            resp = requests.post("http://127.0.0.1:5000/upload-electricity-bill", files=files, timeout=60)
            if resp.ok:
                result = resp.json()
                electricity_units = int(result.get("electricity_units", 0))
                st.success(f"Extracted Units: {electricity_units}")
                if "ocr_preview" in result:
                    with st.expander("Bill OCR Preview"):
                        st.text(result["ocr_preview"])
            else:
                st.error(f"OCR error: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            st.error(f"OCR (bill) error: {e}")

with col2:
    rc_file = st.file_uploader("Vehicle RC (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"], key="rc")
    if rc_file:
        try:
            files = {"file": (rc_file.name, rc_file.getvalue(), rc_file.type or "application/octet-stream")}
            data = {}
            if electricity_units is not None:
                data["units"] = str(electricity_units)
            resp = requests.post("http://127.0.0.1:5000/upload-fuel-doc", files=files, data=data, timeout=60)
            if resp.ok:
                result = resp.json()
                fuel_type = result.get("fuel_type")
                eco_score = result.get("eco_score")
                st.success(f"Fuel Type: {fuel_type or 'Unknown'} | Eco Score: {eco_score if eco_score is not None else 'N/A'}")
                if "ocr_preview" in result:
                    with st.expander("RC OCR Preview"):
                        st.text(result["ocr_preview"])
            else:
                st.error(f"OCR error: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            st.error(f"OCR (RC) error: {e}")

# -------------------- Loan Form --------------------
st.subheader("2) Enter Details")

with st.form("loan_form"):
    colA, colB = st.columns(2)
    with colA:
        # use floats for min_value and default value to avoid mixed types
        loan_amount = st.number_input(
            f"Loan Amount ({currency})",
            min_value=1000.0 if region == "US" else 10000.0,
            value=20000.0 if region == "US" else 200000.0
        )
        term_months = st.selectbox("Loan Term (months)", [36, 60], index=0)
        requested_interest = st.number_input(
            "Requested Interest (%)",
            min_value=1.0,
            max_value=25.0,
            value=12.5
        )

    with colB:
        annual_income = st.number_input(
            f"Annual Income ({currency})",
            min_value=1000.0 if region == "US" else 50000.0,
            value=40000.0 if region == "US" else 600000.0
        )
        monthly_bills = st.number_input(
            f"Monthly Bills ({currency})",
            min_value=0.0,
            value=float(electricity_units) if electricity_units else (1200.0 if region == "US" else 8000.0)
        )

    colC, colD = st.columns(2)
    with colC:
        # years in job is an integer field; keep ints consistent
        emp_length_years = st.number_input("Years in Job", min_value=0, max_value=50, value=5, step=1)
        job_type = st.selectbox("Job Type", ["salaried", "self-employed", "other"])
        gender = st.selectbox("Gender (optional)", ["", "male", "female", "other"])
    with colD:
        past_loans_total_principal = st.number_input(
            f"Past Loans Principal ({currency})",
            min_value=0.0,
            value=10000.0 if region == "US" else 150000.0
        )
        past_loans_late_fee = st.number_input(
            f"Past Loans Late Fee ({currency})",
            min_value=0.0,
            value=50.0 if region == "US" else 500.0
        )
        past_loans_interest = st.number_input(
            f"Past Loans Interest ({currency})",
            min_value=0.0,
            value=500.0 if region == "US" else 7000.0
        )

    st.text_input("Eco Score (auto from RC)", value=str(eco_score) if eco_score is not None else "Not extracted", disabled=True)

    zip_code = st.text_input("ZIP/Postal Code", value="10001" if region == "US" else "560001")
    addr_state = st.text_input("State", value="NY" if region == "US" else "KA")

    submitted = st.form_submit_button("Check Approval")

# -------------------- Send POST request --------------------
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
        "addr_state": addr_state
    }

    try:
        url = "http://127.0.0.1:5000/predict"
        response = requests.post(url, json=payload, timeout=60)

        if not response.ok:
            st.error(f"Prediction API error: {response.status_code} {response.text[:300]}")
        else:
            result = response.json()
            if "approval_probability" not in result:
                st.error(f"Unexpected response: {result}")
            else:
                st.subheader("Prediction Result")
                st.markdown(f"**Region:** {result.get('region', region)}")
                st.markdown(f"**Approval Probability:** {result['approval_probability']*100:.2f}%")
                st.markdown(f"**Decision:** {'✅ Approved' if result['approval_class']==1 else '❌ Rejected'}")
                st.markdown(f"**DTI (display):** {result.get('dti_display', 0):.2f}")

                # Schemes
                schemes = result.get("schemes", [])
                st.subheader("Recommended Schemes")
                if not schemes:
                    st.write("No matching schemes right now.")
                else:
                    for s in schemes:
                        with st.container():
                            st.markdown(f"**{s.get('name','(Unnamed)')}**")
                            st.caption(s.get('description',''))
                            link = s.get('link','')
                            tags = s.get('tags', [])
                            if link:
                                st.write(link)
                            if tags:
                                st.write("Tags:", ", ".join(tags))
                            st.divider()
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
