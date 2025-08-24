# train_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

# --- 1) Load
df = pd.read_csv("data/Eco_Cred_Data.csv")

# --- 2) Encoders (fit & save)
fuel_le = LabelEncoder().fit(df['fuel_type'])
job_le = LabelEncoder().fit(df['job_type'])
loanhist_le = LabelEncoder().fit(df['loan_history'])

df['fuel_type_encoded'] = fuel_le.transform(df['fuel_type'])
df['job_type_encoded'] = job_le.transform(df['job_type'])
df['loan_history_encoded'] = loanhist_le.transform(df['loan_history'])

# --- 3) Features
df['DTI_ratio'] = df['loan_amount'] / df['income']
df['is_EV'] = (df['fuel_type'] == 'Electric').astype(int)
df['high_consumption_flag'] = (df['monthly_units'] > 500).astype(int)
df['eco_category'] = pd.cut(df['eco_score'], bins=[-1, 7, 14, 20], labels=['Low','Medium','High'])
eco_le = LabelEncoder().fit(df['eco_category'])
df['eco_category_encoded'] = eco_le.transform(df['eco_category'])

feature_cols = [
    'income','loan_amount','monthly_units','vehicle_type',
    'fuel_type_encoded','job_type_encoded','loan_history_encoded',
    'credit_score','eco_score','DTI_ratio','is_EV',
    'high_consumption_flag','eco_category_encoded'
]

X = df[feature_cols]
y = df['loan_status']

# --- 4) Train-test split (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5) SMOTE (balance training set)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_res).value_counts().to_dict())

# --- 6) Train tuned RandomForest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
model.fit(X_res, y_res)

# --- 7) Compute best threshold (Youden)
y_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
youden = tpr - fpr
best_thresh = thresholds[np.argmax(youden)]
print("Best threshold (Youden):", best_thresh)

# --- 8) Eval
y_pred = (y_proba >= best_thresh).astype(int)
print("\nClassification Report (tuned):")
print(classification_report(y_test, y_pred))

# --- 9) Save artifacts
joblib.dump(model, "models/loan_model.pkl")
joblib.dump(
    {"fuel_le": fuel_le, "job_le": job_le, "loanhist_le": loanhist_le, "eco_le": eco_le},
    "models/encoders.pkl"
)
joblib.dump(feature_cols, "models/feature_cols.pkl")
joblib.dump(best_thresh, "models/threshold.pkl")

# Save a small background sample for SHAP
shap_bg = X_train.sample(min(200, len(X_train)), random_state=42)
joblib.dump(shap_bg, "models/shap_background.pkl")

print("✅ Saved: loan_model.pkl, encoders.pkl, feature_cols.pkl, threshold.pkl, shap_background.pkl in models/")

