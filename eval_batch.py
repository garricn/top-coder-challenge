import json
import joblib
import pandas as pd
import numpy as np

# Load test cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Load model once
model = joblib.load('reimbursement_model.pkl')

# Prepare input features and expected outputs
inputs = []
expected_outputs = []
bins = [0, 100, 500, 1000, float('inf')]
labels = ['low', 'medium', 'high', 'very_high']
for case in cases:
    trip_duration = float(case['input']['trip_duration_days'])
    miles = float(case['input']['miles_traveled'])
    receipts = float(case['input']['total_receipts_amount'])
    miles_per_day = miles / trip_duration
    daily_receipts = receipts / trip_duration
    receipt_cents = int((receipts % 1) * 100)
    is_5_days = 1 if trip_duration == 5 else 0
    is_49_cents = 1 if receipt_cents == 49 else 0
    is_50_cents = 1 if receipt_cents == 50 else 0
    is_49_or_50_cents = 1 if receipt_cents in [49, 50] else 0
    is_high_receipt = 1 if receipts > 500 else 0  # Adjust threshold if needed
    efficiency_bonus = 1 if 180 <= miles_per_day <= 220 else 0
    low_receipt_flag = 1 if receipts < 50 else 0
    mileage_tier = pd.cut([miles], bins=bins, labels=labels)[0]
    mileage_tier_medium = 1 if mileage_tier == 'medium' else 0
    mileage_tier_high = 1 if mileage_tier == 'high' else 0
    mileage_tier_very_high = 1 if mileage_tier == 'very_high' else 0
    mileage_49_interaction = miles * is_49_cents
    miles_capped = min(miles, 5000)
    receipts_capped = min(receipts, 1000)
    log_total_receipts_amount = np.log1p(receipts)
    log_miles_traveled = np.log1p(miles)
    inputs.append([
        trip_duration, miles, receipts, miles_per_day, daily_receipts, receipt_cents, is_5_days,
        is_49_cents, is_49_or_50_cents, is_high_receipt, efficiency_bonus, low_receipt_flag,
        mileage_49_interaction, mileage_tier_medium, mileage_tier_high, mileage_tier_very_high,
        miles_capped, receipts_capped, log_total_receipts_amount, log_miles_traveled
    ])
    expected_outputs.append(float(case['expected_output']))

columns = [
    'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'miles_per_day',
    'daily_receipts', 'receipt_cents', 'is_5_days', 'is_49_cents', 'is_49_or_50_cents',
    'is_high_receipt', 'efficiency_bonus', 'low_receipt_flag', 'mileage_49_interaction',
    'mileage_tier_medium', 'mileage_tier_high', 'mileage_tier_very_high', 'miles_capped',
    'receipts_capped', 'log_total_receipts_amount', 'log_miles_traveled'
]
X = pd.DataFrame(inputs, columns=columns)
y_true = np.array(expected_outputs)

# Load feature column order from training
try:
    feature_columns = joblib.load('feature_columns.pkl')
    X = X.reindex(columns=feature_columns, fill_value=0)
except Exception as e:
    print(f"Warning: Could not load feature_columns.pkl: {e}")
    # Fallback: use X as is

# Predict all at once
preds = model.predict(X)

# Evaluation (same as eval.sh)
successful_runs = len(preds)
exact_matches = np.sum(np.abs(preds - y_true) < 0.01)
close_matches = np.sum(np.abs(preds - y_true) < 1.0)
errors = np.abs(preds - y_true)
total_error = np.sum(errors)
max_error = np.max(errors)
max_error_idx = np.argmax(errors)

print("🧾 Black Box Challenge - Reimbursement System Evaluation (Python Batch)")
print("=======================================================\n")
print(f"📊 Running evaluation against {len(cases)} test cases...\n")

if successful_runs == 0:
    print("❌ No successful test cases!\n")
    print("Your script either:")
    print("  - Failed to run properly")
    print("  - Produced invalid output format")
    print("  - Timed out on all cases\n")
else:
    avg_error = total_error / successful_runs
    exact_pct = 100 * exact_matches / successful_runs
    close_pct = 100 * close_matches / successful_runs
    print("✅ Evaluation Complete!\n")
    print("📈 Results Summary:")
    print(f"  Total test cases: {len(cases)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print("")
    score = avg_error * 100 + (len(cases) - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print("")
    if exact_matches == len(cases):
        print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
    elif exact_matches > 950:
        print("🥇 Excellent! You are very close to the perfect solution.")
    elif exact_matches > 800:
        print("🥈 Great work! You have captured most of the system behavior.")
    elif exact_matches > 500:
        print("🥉 Good progress! You understand some key patterns.")
    else:
        print("📚 Keep analyzing the patterns in the interviews and test cases.")
    print("")
    print("💡 Tips for improvement:")
    if exact_matches < len(cases):
        print("  Check these high-error cases:")
        # Show top 5 high-error cases
        high_error_indices = np.argsort(errors)[-5:][::-1]
        for idx in high_error_indices:
            c = cases[idx]
            print(f"    Case {idx+1}: {c['input']['trip_duration_days']} days, {c['input']['miles_traveled']} miles, ${c['input']['total_receipts_amount']} receipts")
            print(f"      Expected: ${y_true[idx]:.2f}, Got: ${preds[idx]:.2f}, Error: ${errors[idx]:.2f}")
    print("")

print("📝 Next steps:")
print("  1. Fix any script errors shown above")
print("  2. Ensure your run.sh outputs only a number")
print("  3. Analyze the patterns in the interviews and public cases")
print("  4. Test edge cases around trip length and receipt amounts")
print("  5. Submit your solution via the Google Form when ready!") 