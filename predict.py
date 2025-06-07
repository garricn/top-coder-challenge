import joblib
import sys
import pandas as pd

model = joblib.load('reimbursement_model.pkl')
trip_duration, miles, receipts = map(float, sys.argv[1:4])
miles_per_day = miles / trip_duration
daily_receipts = receipts / trip_duration
receipt_cents = int((receipts % 1) * 100)
is_5_days = 1 if trip_duration == 5 else 0
input_data = pd.DataFrame([[trip_duration, miles, receipts, miles_per_day, daily_receipts, receipt_cents, is_5_days]], 
                          columns=['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day', 'daily_receipts', 'receipt_cents', 'is_5_days'])
prediction = model.predict(input_data)[0]
print(f"{prediction:.2f}")