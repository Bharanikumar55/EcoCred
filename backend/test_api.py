import requests

# URL of your running Flask app
url = 'http://127.0.0.1:5000/predict'

# Minimal user input JSON
data = {
    "loan_amount": 20000,
    "term_months": 36,
    "requested_interest": 12.5,
    "annual_income": 40000,
    "monthly_bills": 1200,
    "emp_length_years": 5,
    "job_type": "salaried",
    "past_loans_total_principal": 10000,
    "past_loans_late_fee": 50,
    "past_loans_interest": 500,
    "eco_score": 0.85,
    "ev_ownership": 1,
    "zip_code": "10001",
    "addr_state": "NY"
}

# Send POST request to Flask API
response = requests.post(url, json=data)

# Print the JSON response
print(response.json())
