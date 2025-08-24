import React, { useState } from "react";
import "./LoanForm.css";

const defaultState = {
  income: 60000,
  loan_amount: 20000,
  monthly_units: 300,
  vehicle_type: 1, // 0=bike, 1=car
  fuel_type: "Petrol",
  credit_score: 700,
  job_type: "Govt",
  loan_history: "Paid"
};

export default function LoanForm() {
  const [form, setForm] = useState(defaultState);
  const [rcFile, setRcFile] = useState(null);
  const [billFile, setBillFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");

  function onChange(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  }

  async function submitManual(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResp(null);

    try {
      const payload = {
        ...form,
        income: Number(form.income),
        loan_amount: Number(form.loan_amount),
        monthly_units: Number(form.monthly_units),
        vehicle_type: Number(form.vehicle_type),
        credit_score: Number(form.credit_score)
      };

      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server error");
      setResp(data);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function submitOCR(e) {
    e.preventDefault();
    if (!rcFile && !billFile) {
      setError("Please upload RC image or Electricity Bill");
      return;
    }

    setLoading(true);
    setError("");
    setResp(null);

    try {
      const formData = new FormData();
      if (rcFile) formData.append("rc_image", rcFile);
      if (billFile) formData.append("bill_image", billFile);

      const res = await fetch("http://127.0.0.1:5000/predict_ocr", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "OCR server error");
      setResp(data);
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="loan-container">
      {/* Manual Section */}
      <div className="form-section">
        <h2>Manual Loan Input</h2>
        <form onSubmit={submitManual}>
          <label>Income (₹)</label>
          <input name="income" value={form.income} onChange={onChange} type="number" />

          <label>Loan amount (₹)</label>
          <input name="loan_amount" value={form.loan_amount} onChange={onChange} type="number" />

          <label>Monthly units (kWh)</label>
          <input name="monthly_units" value={form.monthly_units} onChange={onChange} type="number" />

          <label>Vehicle type</label>
          <select name="vehicle_type" value={form.vehicle_type} onChange={onChange}>
            <option value={0}>Bike</option>
            <option value={1}>Car</option>
          </select>

          <label>Fuel type</label>
          <select name="fuel_type" value={form.fuel_type} onChange={onChange}>
            <option>Electric</option>
            <option>Petrol</option>
            <option>Diesel</option>
          </select>

          <label>Credit score</label>
          <input name="credit_score" value={form.credit_score} onChange={onChange} type="number" />

          <label>Job type</label>
          <select name="job_type" value={form.job_type} onChange={onChange}>
            <option>Govt</option>
            <option>MNC</option>
            <option>Self</option>
          </select>

          <label>Loan history</label>
          <select name="loan_history" value={form.loan_history} onChange={onChange}>
            <option>Paid</option>
            <option>Ongoing</option>
            <option>No History</option>
          </select>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict (Manual)"}
          </button>
        </form>
      </div>

      {/* OCR Section */}
      <div className="form-section">
        <h2>OCR Upload</h2>
        <form onSubmit={submitOCR}>
          <label>Upload RC Image</label>
          <input type="file" accept="image/*,application/pdf" onChange={e => setRcFile(e.target.files[0])} />

          <label>Upload Electricity Bill</label>
          <input type="file" accept="image/*,application/pdf" onChange={e => setBillFile(e.target.files[0])} />

          <button type="submit" disabled={loading}>
            {loading ? "Processing..." : "Predict (OCR)"}
          </button>
        </form>
      </div>

      {/* Result */}
      <div className="result-section">
        {error && <div className="error">{error}</div>}
        {resp && (
          <div className="result-box">
            <h3>Prediction: {resp.prediction === 1 ? "✅ Approved" : "❌ Rejected"}</h3>
            <p><b>Probability:</b> {(resp.probability * 100).toFixed(1)}%</p>

            <h4>Top Reasons</h4>
            <div className="reasons">
              <div>
                <b>Harmful:</b>
                <ul>
                  {resp.reasons?.harmful?.map((r, i) => (
                    <li key={i}>{r.feature}: {r.value} (shap {r.shap_value.toFixed(3)})</li>
                  ))}
                </ul>
              </div>
              <div>
                <b>Helpful:</b>
                <ul>
                  {resp.reasons?.helpful?.map((r, i) => (
                    <li key={i}>{r.feature}: {r.value} (shap {r.shap_value.toFixed(3)})</li>
                  ))}
                </ul>
              </div>
            </div>

            <h4>Recommended Schemes</h4>
            <ul>
              {resp.schemes?.length > 0
                ? resp.schemes.map((s, i) => (
                    <li key={i}><b>{s.name}</b> — {s.description} <a href={s.link} target="_blank" rel="noreferrer">link</a></li>
                  ))
                : <li>No matching schemes</li>}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
