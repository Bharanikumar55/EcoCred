from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, static_folder='static')


# ✅ Load your trained model
model = joblib.load("loan_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        monthly_units = float(request.form['monthly_units'])
        vehicle_type = int(request.form['vehicle_type'])
        eco_score = int(request.form['eco_score'])

        input_data = np.array([[income, loan_amount, monthly_units, vehicle_type, eco_score]])
        prediction = model.predict(input_data)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

