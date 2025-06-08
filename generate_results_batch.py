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
bins = [0, 100, 500, 1000, float('inf')]
labels = ['low', 'medium', 'high', 'very_high']
for case in cases:
    trip_duration = float(case['trip_duration_days'])
    miles = float(case['miles_traveled'])
    receipts = float(case['total_receipts_amount'])
    miles_per_day = miles / trip_duration
    daily_receipts = receipts / trip_duration
    receipt_cents = int((receipts % 1) * 100)
    is_5_days = 1 if trip_duration == 5 else 0
    is_49_cents = 1 if receipt_cents == 49 else 0
    efficiency_bonus = 1 if 180 <= miles_per_day <= 220 else 0
    low_receipt_flag = 1 if receipts < 50 else 0
    mileage_tier = pd.cut([miles], bins=bins, labels=labels)[0]
    mileage_tier_low = 1 if mileage_tier == 'low' else 0
    mileage_tier_medium = 1 if mileage_tier == 'medium' else 0
    mileage_tier_very_high = 1 if mileage_tier == 'very_high' else 0
    inputs.append([
        trip_duration, miles, receipts, miles_per_day, daily_receipts, receipt_cents, is_5_days,
        is_49_cents, efficiency_bonus, low_receipt_flag,
        mileage_tier_low, mileage_tier_medium, mileage_tier_very_high
    ])

X = pd.DataFrame(inputs, columns=[
    'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
    'miles_per_day', 'daily_receipts', 'receipt_cents', 'is_5_days',
    'is_49_cents', 'efficiency_bonus', 'low_receipt_flag',
    'mileage_tier_low', 'mileage_tier_medium', 'mileage_tier_very_high'])

# Predict all at once
preds = model.predict(X)

# Write results to private_results.txt
with open('private_results.txt', 'w') as f:
    for pred in preds:
        f.write(f"{pred:.2f}\n") 