import pandas as pd
import random

# Load your original dataset
df = pd.read_csv("clean_midoriloan_data.csv")


# Add new columns using random or calculated logic
job_types = ['MNC', 'Govt', 'Self']
loan_histories = ['Paid', 'Ongoing', 'No History']
fuel_types = ['Electric', 'Petrol', 'Diesel']

# Randomly assign values
df['job_type'] = [random.choice(job_types) for _ in range(len(df))]
df['loan_history'] = [random.choice(loan_histories) for _ in range(len(df))]
df['fuel_type'] = [random.choice(fuel_types) for _ in range(len(df))]

# Generate credit scores based on loan history
def generate_credit_score(history):
    if history == 'Paid':
        return random.randint(700, 800)
    elif history == 'Ongoing':
        return random.randint(600, 690)
    else:
        return random.randint(650, 730)

df['credit_score'] = df['loan_history'].apply(generate_credit_score)

# Recalculate eco_score based on fuel and monthly usage
def calculate_eco_score(fuel_type, monthly_units):
    fuel_points = {'Electric': 2, 'Petrol': 1, 'Diesel': 0}
    usage_score = 2 if monthly_units <= 200 else 1 if monthly_units <= 400 else 0
    return min(2, fuel_points.get(fuel_type, 0) + usage_score)

df['eco_score'] = df.apply(lambda row: calculate_eco_score(row['fuel_type'], row['monthly_units']), axis=1)

# Save updated dataset
df.to_csv("updated_midoriloan_data.csv", index=False)

print("✅ Dataset updated and saved as 'updated_midoriloan_data.csv'")
