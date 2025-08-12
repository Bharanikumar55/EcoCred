from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import (
    preprocess_input,
    get_model_and_explainer,
    shap_reason_engine,
    generate_chatbot_message,
    recommend_schemes
)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests for frontend

# Load model, explainer, and configs once at startup
model, explainer, threshold, feature_cols = get_model_and_explainer()

# ✅ Home route for testing
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "success",
        "message": "EcoCred backend is running. Use POST /predict or /chat."
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_data = request.get_json()
        if not raw_data:
            return jsonify({"error": "No input data provided"}), 400

        # Preprocess user input
        X_user = preprocess_input(raw_data)

        # Prediction
        pred_proba = model.predict_proba(X_user)[0][1]
        prediction = int(pred_proba >= threshold)

        # SHAP reasoning
        reason_json = shap_reason_engine(X_user, explainer, class_index=prediction)
        chatbot_msg = generate_chatbot_message(reason_json)

        # Scheme recommendations
        user_row = raw_data.copy()
        user_row.update(X_user.iloc[0].to_dict())
        schemes = recommend_schemes(user_row)

        return jsonify({
            "probability": float(pred_proba),
            "prediction": prediction,
            "reasons": reason_json,
            "chatbot_message": chatbot_msg,
            "schemes": schemes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data["message"].lower()

        # Simple rule-based responses
        if "loan" in user_message and "approve" in user_message:
            reply = "To get your loan approved, focus on improving your credit score, lowering your DTI ratio, and providing complete documentation."
        elif "scheme" in user_message:
            dummy_user_data = {
                "eco_score": 15,
                "is_EV": 1,
                "income": 90000,
                "loan_amount": 600000,
                "job_type_encoded": 0
            }
            schemes = recommend_schemes(dummy_user_data)
            if schemes:
                reply = "Here are some schemes you may qualify for:\n" + "\n".join(
                    [f"- {s['name']}: {s['link']}" for s in schemes]
                )
            else:
                reply = "I couldn't find any matching schemes for you right now."
        else:
            reply = "I'm still learning! Please ask about loans, approvals, or schemes."

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Host set to 0.0.0.0 so it’s accessible in Docker or external devices
    app.run(debug=True, host="0.0.0.0", port=5000)
