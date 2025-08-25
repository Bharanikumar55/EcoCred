# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# -------------------------------
import joblib


# Use raw string for Windows path
model_path = r'C:\Users\jites\Desktop\EcoCred\backend\models\lending_club_model_1.pkl'
import os

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
    'A1':0,'A2':1,'A3':2,'A4':3,'A5':4,
    'B1':5,'B2':6,'B3':7,'B4':8,'B5':9,
    'C1':10,'C2':11,'C3':12,'C4':13,'C5':14
    # extend if needed
}

# -------------------------------
# Helper: Calculate derived features
# -------------------------------
def calculate_features(df):
    # Loan related
    rate = df['requested_interest']/100
    term = df['term_months']
    df['installment'] = df['loan_amount'] * (rate/12) / (1 - (1 + rate/12)**(-term))
    df['out_prncp'] = df['loan_amount']  # initial outstanding principal
    df['out_prncp_inv'] = df['loan_amount']  # for investors
    
    # Income & debt
    df['dti'] = df['monthly_bills'] / (df['annual_income']/12)
    df['revol_util_num'] = df.get('revol_util', 0)  # optional, default 0
    
    # Employment
    df['emp_length_years'] = df['emp_length_years']
    df['job_type_code'] = df['job_type'].map(job_type_mapping)
    
    # Past loans
    df['total_rec_prncp'] = df.get('past_loans_total_principal', 0)
    df['total_rec_late_fee'] = df.get('past_loans_late_fee', 0)
    df['total_rec_int'] = df.get('past_loans_interest', 0)
    df['credit_history_months'] = df.get('credit_history_months', df['emp_length_years']*12)
    df['total_pymnt'] = df['total_rec_prncp'] + df['total_rec_int'] + df['total_rec_late_fee']
    df['total_pymnt_inv'] = df['total_pymnt']
    df['recoveries'] = df.get('recoveries',0)
    
    # FICO and sub-grade
    df['fico_range_low'] = df.get('fico_range_low', 600)
    df['fico_range_high'] = df.get('fico_range_high', 700)
    df['sub_grade_code'] = df.get('sub_grade','A3')
    df['sub_grade_code'] = df['sub_grade_code'].map(sub_grade_mapping)
    
    # Eco / Green
    df['eco_score'] = df.get('eco_score',0.5)
    df['is_EV'] = df.get('ev_ownership',0)
    
    # Term months
    df['term_months'] = df['term_months']
    
    # Other numeric features defaulted if not provided
    df['total_acc'] = df.get('total_acc', 5)
    df['open_acc'] = df.get('open_acc', 3)
    df['mths_since_last_delinq'] = df.get('mths_since_last_delinq', 24)
    df['last_pymnt_amnt'] = df.get('last_pymnt_amnt', df['installment'])
    df['last_pymnt_d'] = df.get('last_pymnt_d', '2025-08-01')
    df['last_credit_pull_d'] = df.get('last_credit_pull_d', '2025-08-01')
    df['addr_state'] = df.get('addr_state','NY')
    df['zip_code'] = df.get('zip_code','10001')
    
    # Interest numeric
    df['int_rate_num'] = df['requested_interest']
    
    return df

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        
        # Calculate derived features
        df = calculate_features(df)
        
        # Select features used by the model (example: same as your top 30)
        features = [
            'total_rec_prncp', 'last_pymnt_d', 'last_pymnt_amnt', 'loan_amount', 'installment',
            'out_prncp', 'last_fico_range_high', 'total_rec_late_fee', 'total_rec_int',
            'credit_history_months', 'total_pymnt', 'dti', 'last_credit_pull_d', 'zip_code',
            'revol_util_num', 'annual_income', 'total_pymnt_inv', 'recoveries', 'revol_bal',
            'mths_since_last_delinq', 'int_rate_num', 'out_prncp_inv', 'open_acc', 'addr_state',
            'fico_range_low', 'total_acc', 'emp_length_years', 'sub_grade_code', 'term_months', 'last_fico_range_low'
        ]
        
        # Fill missing features with 0 or median if needed
        for f in features:
            if f not in df.columns:
                df[f] = 0
        
        X = df[features]
        
        # Ensure numeric columns are float
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Prediction
        prob_default = model.predict_proba(X)[:,1][0]
        approval_prob = 1 - prob_default
        approval_class = int(approval_prob >= 0.5)
        
        # Simple recommendations
        recommendations = []
        if data.get('eco_score',0) > 0.8:
            recommendations.append('Green Loan / Reduced Interest')
        if data.get('annual_income',0) < 25000:
            recommendations.append('PM Mudra Loan')
        if data.get('loan_amount',0) > 50000:
            recommendations.append('Check SBI Green Home Loan')
        
        return jsonify({
            'approval_probability': round(approval_prob,4),
            'approval_class': approval_class,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

# -------------------------------
# Run app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
