import joblib
import sys
import pandas as pd

model = joblib.load('reimbursement_model.pkl')
trip_duration, miles, receipts = map(float, sys.argv[1:4])
miles_per_day = miles / trip_duration
daily_receipts = receipts / trip_duration
receipt_cents = int((receipts % 1) * 100)
is_5_days = 1 if trip_duration == 5 else 0
is_49_cents = 1 if receipt_cents == 49 else 0
efficiency_bonus = 1 if 180 <= miles_per_day <= 220 else 0
low_receipt_flag = 1 if receipts < 50 else 0
# Mileage tier dummies (only low, medium, very_high)
bins = [0, 100, 500, 1000, float('inf')]
labels = ['low', 'medium', 'high', 'very_high']
mileage_tier = pd.cut([miles], bins=bins, labels=labels)[0]
mileage_tier_low = 1 if mileage_tier == 'low' else 0
mileage_tier_medium = 1 if mileage_tier == 'medium' else 0
mileage_tier_very_high = 1 if mileage_tier == 'very_high' else 0
# Assemble features in correct order
input_data = pd.DataFrame([[
    trip_duration, miles, receipts, miles_per_day, daily_receipts, receipt_cents, is_5_days,
    is_49_cents, efficiency_bonus, low_receipt_flag,
    mileage_tier_low, mileage_tier_medium, mileage_tier_very_high
]], columns=[
    'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'daily_receipts', 'receipt_cents', 'is_5_days',
    'is_49_cents', 'efficiency_bonus', 'low_receipt_flag',
    'mileage_tier_low', 'mileage_tier_medium', 'mileage_tier_very_high'])
prediction = model.predict(input_data)[0]
print(f"{prediction:.2f}")