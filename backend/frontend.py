import streamlit as st
import requests

st.set_page_config(page_title="EcoCred Loan Prediction", layout="centered")
st.title("EcoCred Loan Approval Predictor")

st.write("Upload your documents (JPG/PNG) and enter details below:")

# -------------------- File Uploads for OCR --------------------
eco_score = None
electricity_units = None
fuel_type = None

# Electricity Bill Upload (image only for now)
elec_file = st.file_uploader("Upload Electricity Bill (jpg/png)", type=["jpg", "jpeg", "png"])
if elec_file:
    try:
        # Build multipart file tuple: (filename, bytes, content_type)
        files = {"file": (elec_file.name, elec_file.getvalue(), elec_file.type or "image/jpeg")}
        resp = requests.post("http://127.0.0.1:5000/upload-electricity-bill", files=files, timeout=30)
        if resp.ok:
            result = resp.json()
            if "electricity_units" in result:
                electricity_units = int(result["electricity_units"])
                st.success(f"✅ Extracted Units: {electricity_units}")
            else:
                st.warning("⚠️ Could not extract units from bill")
            # Show optional OCR preview text if backend returns it
            if "ocr_preview" in result:
                st.text_area("OCR preview (bill)", result["ocr_preview"], height=120)
        else:
            st.error(f"OCR request failed (bill): {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        st.error(f"OCR error (bill): {e}")

# RC Upload (image only for now)
rc_file = st.file_uploader("Upload Vehicle RC (jpg/png)", type=["jpg", "jpeg", "png"])
if rc_file:
    try:
        files = {"file": (rc_file.name, rc_file.getvalue(), rc_file.type or "image/jpeg")}
        # Optionally include units so backend can compute combined eco_score; send as form data
        data = {}
        if electricity_units:
            data["units"] = str(electricity_units)
        resp = requests.post("http://127.0.0.1:5000/upload-fuel-doc", files=files, data=data, timeout=30)
        if resp.ok:
            result = resp.json()
            if "fuel_type" in result:
                fuel_type = result["fuel_type"]
                eco_score = result.get("eco_score", None)
                st.success(f"✅ Fuel Type: {fuel_type}" + (f", Eco Score: {eco_score}" if eco_score is not None else ""))
            else:
                st.warning("⚠️ Could not extract fuel type")
            if "ocr_preview" in result:
                st.text_area("OCR preview (RC)", result["ocr_preview"], height=120)
        else:
            st.error(f"OCR request failed (RC): {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        st.error(f"OCR error (RC): {e}")

# -------------------- Loan Form --------------------
with st.form("loan_form"):
    # Loan info
    loan_amount = st.number_input("Loan Amount", min_value=1000, value=20000)
    term_months = st.selectbox("Loan Term (months)", [36, 60])
    requested_interest = st.number_input("Requested Interest (%)", min_value=1.0, max_value=25.0, value=12.5)

    # Income & debt
    annual_income = st.number_input("Annual Income", min_value=1000, value=40000)
    monthly_bills = st.number_input(
        "Monthly Bills",
        min_value=0,
        value=electricity_units if electricity_units else 1200
    )

    # Employment
    emp_length_years = st.number_input("Years in Job", min_value=0, max_value=50, value=5)
    job_type = st.selectbox("Job Type", ["salaried", "self-employed", "other"])

    # Past loans
    past_loans_total_principal = st.number_input("Past Loans Principal", min_value=0, value=10000)
    past_loans_late_fee = st.number_input("Past Loans Late Fee", min_value=0, value=50)
    past_loans_interest = st.number_input("Past Loans Interest", min_value=0, value=500)

    # Auto eco_score from RC (display only)
    st.text_input("Eco Score (Auto)", value=str(eco_score) if eco_score is not None else "Not extracted", disabled=True)

    # Location
    zip_code = st.text_input("ZIP Code", value="10001")
    addr_state = st.text_input("State", value="NY")

    submitted = st.form_submit_button("Check Approval")

# -------------------- Send POST request --------------------
if submitted:
    # Basic validation
    if rc_file is None or elec_file is None:
        st.warning("Please upload both an electricity bill image and a vehicle RC image for best results.")
    else:
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
            "eco_score": eco_score if eco_score is not None else 0.5,  # fallback
            "fuel_type": fuel_type if fuel_type else "Unknown",
            "zip_code": zip_code,
            "addr_state": addr_state
        }

        try:
            url = "http://127.0.0.1:5000/predict"
            response = requests.post(url, json=input_data, timeout=30)

            if not response.ok:
                st.error(f"Prediction API error: {response.status_code} {response.text[:200]}")
            else:
                result = response.json()
                # Validate result structure
                if "approval_probability" not in result:
                    st.error(f"Unexpected response from API: {result}")
                else:
                    st.subheader("Prediction Result")
                    st.write(f"**Approval Probability:** {result['approval_probability']*100:.2f}%")
                    st.write(f"**Approval Class:** {'✅ Approved' if result['approval_class']==1 else '❌ Rejected'}")
                    st.write(f"**Recommendations:** {', '.join(result['recommendations']) if result['recommendations'] else 'None'}")

        except Exception as e:
            st.error(f"Error connecting to API: {e}")
