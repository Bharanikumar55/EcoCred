import os
import joblib
import pandas as pd
import numpy as np
import shap

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Load artifacts ---
model = joblib.load(os.path.join(MODELS_DIR, "loan_model.pkl"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
threshold = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
shap_background = joblib.load(os.path.join(MODELS_DIR, "shap_background.pkl"))

fuel_le = encoders['fuel_le']
job_le = encoders['job_le']
loanhist_le = encoders['loanhist_le']
eco_le = encoders['eco_le']

# --- Preprocessing ---
def preprocess_input(raw):
    df = pd.DataFrame([raw])

    df['fuel_type_encoded'] = fuel_le.transform(df['fuel_type'])
    df['job_type_encoded'] = job_le.transform(df['job_type'])
    df['loan_history_encoded'] = loanhist_le.transform(df['loan_history'])

    df['DTI_ratio'] = df['loan_amount'] / df['income']
    df['is_EV'] = (df['fuel_type'] == 'Electric').astype(int)
    df['high_consumption_flag'] = (df['monthly_units'] > 500).astype(int)
    df['eco_category'] = pd.cut(
        df['eco_score'], bins=[-1, 7, 14, 20],
        labels=['Low', 'Medium', 'High']
    )

    try:
        df['eco_category_encoded'] = eco_le.transform(df['eco_category'])
    except ValueError:
        new_classes = np.unique(np.concatenate((eco_le.classes_, df['eco_category'].unique())))
        eco_le.classes_ = new_classes
        df['eco_category_encoded'] = eco_le.transform(df['eco_category'])

    return pd.DataFrame(df[feature_cols], columns=feature_cols)

# --- SHAP explainer ---
def get_model_and_explainer():
    explainer = shap.Explainer(model, shap_background)
    return model, explainer, threshold, feature_cols

def extract_shap_for_user(shap_values_obj, class_index=0):
    arr = shap_values_obj.values
    if arr.ndim == 3:
        return arr[0, :, class_index]
    elif arr.ndim == 2:
        return arr[0, :]
    else:
        raise ValueError(f"Unexpected shap array shape: {arr.shape}")

def shap_reason_engine(X_user_df, explainer, class_index=0):
    shap_vals_obj = explainer(X_user_df)
    shap_vals = extract_shap_for_user(shap_vals_obj, class_index)
    features = np.array(X_user_df.columns)
    values = np.ravel(X_user_df.values)
    df = pd.DataFrame({
        "feature": features,
        "value": values,
        "shap_value": shap_vals
    }).sort_values(by="shap_value", ascending=True)

    harmful = df.head(3).to_dict(orient="records")
    helpful = df.tail(3).to_dict(orient="records")
    decision = "Rejected" if class_index == 0 else "Approved"
    return {"decision": decision, "harmful": harmful, "helpful": helpful}

def generate_chatbot_message(reason_json):
    decision = reason_json['decision']
    out_lines = []
    if decision == "Rejected":
        out_lines.append(f"❌ Your loan application was {decision}.")
        out_lines.append("Top reasons:")
        for r in reason_json['harmful']:
            out_lines.append(f"- {r['feature']}: {r['value']} (negative impact)")
        out_lines.append("\n💡 Suggestions to improve:")
        for r in reason_json['helpful']:
            out_lines.append(f"- Improve {r['feature']} (current {r['value']})")
    else:
        out_lines.append(f"✅ Your loan application was {decision}.")
        out_lines.append("Top supporting factors:")
        for r in reason_json['helpful']:
            out_lines.append(f"- {r['feature']}: {r['value']}")
    return "\n".join(out_lines)

# --- Schemes ---
SCHEMES = [
    {
        "name": "PM-KUSUM Solar Pump Scheme",
        "criteria": lambda row: row["eco_score"] >= 15 and row["is_EV"] == 1,
        "description": "For farmers adopting renewable energy and electric equipment.",
        "link": "https://mnre.gov.in/pmkusum"
    },
    {
        "name": "SBI Green Home Loan",
        "criteria": lambda row: row["eco_score"] >= 12 and row["loan_amount"] > 500000,
        "description": "For homes with energy-efficient designs and materials.",
        "link": "https://sbi.co.in/green-homes"
    },
    {
        "name": "MUDRA Yojana",
        "criteria": lambda row: row["income"] <= 100000 and row["job_type_encoded"] in [0, 1],
        "description": "Micro-loans for small businesses and entrepreneurs.",
        "link": "https://mudra.org.in"
    },
    {
        "name": "EV Purchase Subsidy",
        "criteria": lambda row: row["is_EV"] == 1,
        "description": "Government subsidy for purchasing electric vehicles.",
        "link": "https://fame2.in"
    }
]

def recommend_schemes(user_row):
    recommendations = []
    for scheme in SCHEMES:
        try:
            if scheme["criteria"](user_row):
                recommendations.append({
                    "name": scheme["name"],
                    "description": scheme["description"],
                    "link": scheme["link"]
                })
        except KeyError:
            continue
    return recommendations
