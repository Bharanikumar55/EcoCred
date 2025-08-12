import React, { useState } from "react";

const defaultState = {
  income: 60000,
  loan_amount: 20000,
  monthly_units: 300,
  vehicle_type: 0, // 0=bike,1=car
  fuel_type: "Petrol", // Electric, Petrol, Diesel
  eco_score: 10,
  credit_score: 700,
  job_type: "Govt", // Govt, MNC, Self
  loan_history: "No History" // Paid, Ongoing, No History
};

export default function LoanForm(){
  const [form, setForm] = useState(defaultState);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");

  function onChange(e){
    const { name, value } = e.target;
    setForm(prev => ({...prev, [name]: value}));
  }

  async function submit(e){
    e?.preventDefault();
    setLoading(true); setError(""); setResp(null);
    try{
      // convert numeric fields
      const payload = {
        ...form,
        income: Number(form.income),
        loan_amount: Number(form.loan_amount),
        monthly_units: Number(form.monthly_units),
        vehicle_type: Number(form.vehicle_type),
        eco_score: Number(form.eco_score),
        credit_score: Number(form.credit_score)
      };

      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if(!res.ok) throw new Error(data.error || "Server error");
      setResp(data);
    }catch(err){
      setError(err.message || "Request failed");
    }finally{
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2>Manual Loan Input</h2>
      <p className="small">Fill user details below and click Predict</p>

      <form onSubmit={submit}>
        <label className="label">Income (₹)</label>
        <input className="field" name="income" value={form.income} onChange={onChange} type="number" />

        <label className="label">Loan amount (₹)</label>
        <input className="field" name="loan_amount" value={form.loan_amount} onChange={onChange} type="number" />

        <div className="row">
          <div className="col">
            <label className="label">Monthly units (kWh)</label>
            <input className="field" name="monthly_units" value={form.monthly_units} onChange={onChange} type="number" />
          </div>
          <div className="col">
            <label className="label">Vehicle type</label>
            <select className="field" name="vehicle_type" value={form.vehicle_type} onChange={onChange}>
              <option value={0}>Bike</option>
              <option value={1}>Car</option>
            </select>
          </div>
        </div>

        <label className="label">Fuel type</label>
        <select className="field" name="fuel_type" value={form.fuel_type} onChange={onChange}>
          <option>Electric</option>
          <option>Petrol</option>
          <option>Diesel</option>
        </select>

        <div className="row">
          <div className="col">
            <label className="label">Eco score (0–20)</label>
            <input className="field" name="eco_score" value={form.eco_score} onChange={onChange} type="number" min="0" max="20" />
          </div>
          <div className="col">
            <label className="label">Credit score</label>
            <input className="field" name="credit_score" value={form.credit_score} onChange={onChange} type="number" />
          </div>
        </div>

        <label className="label">Job type</label>
        <select className="field" name="job_type" value={form.job_type} onChange={onChange}>
          <option>Govt</option><option>MNC</option><option>Self</option>
        </select>

        <label className="label">Loan history</label>
        <select className="field" name="loan_history" value={form.loan_history} onChange={onChange}>
          <option>Paid</option><option>Ongoing</option><option>No History</option>
        </select>

        <div style={{marginTop:12}}>
          <button className="btn" type="submit" disabled={loading}>{loading ? "Predicting..." : "Predict"}</button>
        </div>
      </form>

      {error && <div style={{color:"#ffb4b4", marginTop:10}}>{error}</div>}

      {resp && (
        <div className="result-box">
          <strong>Prediction:</strong> {resp.prediction === 1 ? "✅ Approved" : "❌ Rejected"} <br/>
          <strong>Probability:</strong> {(resp.probability*100).toFixed(1)}% <br/>

          <div style={{marginTop:10}}>
            <strong>Top reasons (harmful):</strong>
            <table className="table">
              <thead><tr><th>feature</th><th>value</th><th>shap</th></tr></thead>
              <tbody>
                {resp.reasons?.harmful?.map((r,i)=>(
                  <tr key={i}>
                    <td>{r.feature}</td>
                    <td>{String(r.value)}</td>
                    <td>{Number(r.shap_value).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <strong>Top supports (helpful):</strong>
            <table className="table">
              <tbody>
                {resp.reasons?.helpful?.map((r,i)=>(
                  <tr key={i}>
                    <td>{r.feature}</td>
                    <td>{String(r.value)}</td>
                    <td>{Number(r.shap_value).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{marginTop:10}}>
            <strong>Recommended Schemes:</strong>
            <ul>
              {(resp.schemes || []).length === 0 && <li className="small">No matching schemes</li>}
              {(resp.schemes || []).map((s,i)=>(
                <li key={i}><b>{s.name}</b> — {s.description} <a href={s.link} target="_blank" rel="noreferrer">link</a></li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
