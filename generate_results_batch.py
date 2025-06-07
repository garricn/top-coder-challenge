import json
import joblib
import pandas as pd

# Load private test cases
with open('private_cases.json', 'r') as f:
    cases = json.load(f)

# Load model once
model = joblib.load('reimbursement_model.pkl')

# Prepare input features
inputs = []
for case in cases:
    trip_duration = float(case['trip_duration_days'])
    miles = float(case['miles_traveled'])
    receipts = float(case['total_receipts_amount'])
    miles_per_day = miles / trip_duration
    daily_receipts = receipts / trip_duration
    receipt_cents = int((receipts % 1) * 100)
    is_5_days = 1 if trip_duration == 5 else 0
    inputs.append([
        trip_duration, miles, receipts, miles_per_day, daily_receipts, receipt_cents, is_5_days
    ])

X = pd.DataFrame(inputs, columns=[
    'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
    'miles_per_day', 'daily_receipts', 'receipt_cents', 'is_5_days'])

# Predict all at once
preds = model.predict(X)

# Write results to private_results.txt
with open('private_results.txt', 'w') as f:
    for pred in preds:
        f.write(f"{pred:.2f}\n") 