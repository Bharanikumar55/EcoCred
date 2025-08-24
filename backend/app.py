import os
import joblib
import shap
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# utils helpers
from utils import preprocess_input, recommend_schemes
from ocr_utils import extract_ocr_data   # keep OCR separate

app = Flask(__name__)
CORS(app)

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Load artifacts ---
model = joblib.load(os.path.join(MODELS_DIR, "loan_model.pkl"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
threshold = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
shap_background = joblib.load(os.path.join(MODELS_DIR, "shap_background.pkl"))

explainer = shap.TreeExplainer(model)

# Store last prediction for chatbot context
last_prediction = {"decision": None, "probability": None, "reasons": None, "schemes": None}


# ---------------------------
# Manual Predict Route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    global last_prediction
    data = request.get_json()

    try:
        # prepare features using utils
        X = preprocess_input(data, encoders, feature_cols)

        # model prediction
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= threshold)

        # SHAP values
        shap_values = explainer.shap_values(X)
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        df_shap = pd.DataFrame({
            "feature": feature_cols,
            "value": X.iloc[0].values,
            "shap_value": vals
        })

        helpful = df_shap.sort_values("shap_value", ascending=False).head(3).to_dict(orient="records")
        harmful = df_shap.sort_values("shap_value").head(3).to_dict(orient="records")

        # recommend schemes
        schemes = recommend_schemes(data)

        result = {
            "prediction": pred,
            "probability": proba,
            "reasons": {"helpful": helpful, "harmful": harmful},
            "schemes": schemes
        }

        # store for chatbot use
        last_prediction = {
            "decision": "Approved" if pred == 1 else "Rejected",
            "probability": round(proba, 3),
            "reasons": {"helpful": helpful, "harmful": harmful},
            "schemes": schemes
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------
# OCR Predict Route
# ---------------------------
@app.route("/predict-ocr", methods=["POST"])
def predict_ocr():
    global last_prediction
    if "rc" not in request.files and "bill" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    extracted = {}
    if "rc" in request.files:
        rc_file = request.files["rc"]
        path_rc = os.path.join(BASE_DIR, "temp_rc.jpg")
        rc_file.save(path_rc)
        extracted.update(extract_ocr_data(path_rc))

    if "bill" in request.files:
        bill_file = request.files["bill"]
        path_bill = os.path.join(BASE_DIR, "temp_bill.jpg")
        bill_file.save(path_bill)
        extracted.update(extract_ocr_data(path_bill))

    # auto-fill defaults if missing
    user_input = {
        "income": 60000,
        "loan_amount": 20000,
        "vehicle_type": 1,  # assume car
        "fuel_type": extracted.get("fuel_type", "Petrol"),
        "eco_score": 10,
        "credit_score": 650,
        "job_type": "Govt",
        "loan_history": "No History",
        "monthly_units": extracted.get("monthly_units", 300)
    }

    try:
        # prepare features
        X = preprocess_input(user_input, encoders, feature_cols)

        # prediction
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= threshold)

        # SHAP
        shap_values = explainer.shap_values(X)
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        df_shap = pd.DataFrame({
            "feature": feature_cols,
            "value": X.iloc[0].values,
            "shap_value": vals
        })

        helpful = df_shap.sort_values("shap_value", ascending=False).head(3).to_dict(orient="records")
        harmful = df_shap.sort_values("shap_value").head(3).to_dict(orient="records")

        schemes = recommend_schemes(user_input)

        result = {
            "prediction": pred,
            "probability": proba,
            "ocr_data": extracted,
            "reasons": {"helpful": helpful, "harmful": harmful},
            "schemes": schemes
        }

        # store for chatbot
        last_prediction = {
            "decision": "Approved" if pred == 1 else "Rejected",
            "probability": round(proba, 3),
            "reasons": {"helpful": helpful, "harmful": harmful},
            "schemes": schemes
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------
# Chatbot Route
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    global last_prediction
    data = request.get_json()
    user_msg = data.get("message", "")

    if not user_msg:
        return jsonify({"reply": "⚠️ Please type something."})

    user_msg = user_msg.lower()

    if "why" in user_msg and last_prediction["decision"]:
        reply = f"📊 Last decision: *{last_prediction['decision']}* (prob={last_prediction['probability']}).\n\n"
        reply += "✅ Helpful factors:\n"
        for r in last_prediction["reasons"]["helpful"]:
            reply += f"- {r['feature']} = {r['value']} (shap={r['shap_value']:.3f})\n"
        reply += "\n❌ Harmful factors:\n"
        for r in last_prediction["reasons"]["harmful"]:
            reply += f"- {r['feature']} = {r['value']} (shap={r['shap_value']:.3f})\n"
    elif "scheme" in user_msg and last_prediction["schemes"]:
        reply = "💡 Recommended schemes:\n"
        for s in last_prediction["schemes"]:
            reply += f"- {s['name']} — {s['description']}\n"
    else:
        reply = "🤖 Ask me about *eligibility*, *schemes*, or *reasons* for loan decisions."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
