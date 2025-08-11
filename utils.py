# utils.py
import joblib
import pandas as pd
import numpy as np
import shap

# load artifacts
model = joblib.load("loan_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_cols = joblib.load("feature_cols.pkl")
threshold = joblib.load("threshold.pkl")
shap_bg = joblib.load("shap_background.pkl")  # DataFrame for explainer background

fuel_le = encoders['fuel_le']
job_le = encoders['job_le']
loanhist_le = encoders['loanhist_le']
eco_le = encoders['eco_le']

def preprocess_input(raw):
    """
    raw: dict with keys = original raw columns:
      income, loan_amount, monthly_units, vehicle_type (0/1), fuel_type, eco_score,
      credit_score, job_type, loan_history
    returns: pd.DataFrame with columns in feature_cols order
    """
    # Always make a DataFrame
    df = pd.DataFrame([raw])

    # Encode categorical features
    df['fuel_type_encoded'] = fuel_le.transform(df['fuel_type'])
    df['job_type_encoded'] = job_le.transform(df['job_type'])
    df['loan_history_encoded'] = loanhist_le.transform(df['loan_history'])

    # Derived features
    df['DTI_ratio'] = df['loan_amount'] / df['income']
    df['is_EV'] = (df['fuel_type'] == 'Electric').astype(int)
    df['high_consumption_flag'] = (df['monthly_units'] > 500).astype(int)
    df['eco_category'] = pd.cut(df['eco_score'], bins=[-1, 7, 14, 20],
                                labels=['Low', 'Medium', 'High'])

    # Encode eco_category safely
    try:
        df['eco_category_encoded'] = eco_le.transform(df['eco_category'])
    except ValueError:
        import numpy as np
        # Merge missing labels into encoder
        new_classes = np.unique(np.concatenate((eco_le.classes_, df['eco_category'].unique())))
        eco_le.classes_ = new_classes
        df['eco_category_encoded'] = eco_le.transform(df['eco_category'])

    # Ensure correct order of features and correct shape
    X_user = df[feature_cols]
    return pd.DataFrame(X_user, columns=feature_cols)


def get_model_and_explainer():
    # create a shap Explainer at app startup (fast for single-explains)
    explainer = shap.Explainer(model, shap_bg)
    return model, explainer, threshold, feature_cols

def extract_shap_for_user(shap_values_obj, class_index=0):
    """
    shap_values_obj: output of explainer(X_user_df)
    returns a 1D array of length n_features for chosen class (0=Rejected,1=Approved)
    robust to shapes.
    """
    arr = shap_values_obj.values
    # arr could be shape (1, n_features) or (1, n_features, n_classes)
    if arr.ndim == 3:
        # pick class index
        return arr[0, :, class_index]
    elif arr.ndim == 2:
        return arr[0, :]
    else:
        raise ValueError("Unexpected shap array shape: " + str(arr.shape))

def shap_reason_engine(X_user_df, explainer, class_index=0):
    """
    X_user_df: a 1-row DataFrame
    returns dict with decision, harmful list, helpful list
    """
    shap_vals_obj = explainer(X_user_df)
    shap_vals = extract_shap_for_user(shap_vals_obj, class_index=class_index)
    features = np.array(X_user_df.columns)
    values = np.ravel(X_user_df.values)
    df = pd.DataFrame({"feature": features, "value": values, "shap_value": shap_vals})
    df = df.sort_values(by="shap_value", ascending=True)  # harmful (neg) first
    harmful = df.head(3).to_dict(orient="records")
    helpful = df.tail(3).to_dict(orient="records")
    decision = "Rejected" if class_index == 0 else "Approved"
    return {"decision": decision, "harmful": harmful, "helpful": helpful}

def generate_chatbot_message(reason_json):
    """Turn reason_json into a friendly string"""
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

# utils.py

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
        "criteria": lambda row: row["income"] <= 100000 and row["job_type_encoded"] in [0, 1],  # Self/Small business
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
