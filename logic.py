import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

# Load enhanced dataset
inputs = pd.read_csv('/content/enhanced_inputs_v3.csv')
outputs = pd.read_csv('/content/enhanced_outputs.csv')['expected_output']

# Refine features based on plot and high-error cases
inputs['miles_capped'] = np.where(inputs['miles_traveled'] > 1000, 1000, inputs['miles_traveled'])  # Cap high mileage
inputs['receipts_capped'] = np.where(inputs['total_receipts_amount'] > 1000, 1000, inputs['total_receipts_amount'])  # Cap high receipts
inputs['is_high_receipt'] = (inputs['total_receipts_amount'] > 1000).astype(int)  # Flag high receipts
inputs['mileage_49_interaction'] = inputs['miles_traveled'] * inputs['is_49_or_50_cents']  # Interaction for 49-cent quirk

# Update mileage tiers for better granularity
inputs['mileage_tier'] = pd.cut(inputs['miles_traveled'], bins=[0, 100, 500, 1000, 1500, float('inf')], 
                                labels=['low', 'medium', 'high', 'very_high', 'ultra_high'])

# Convert categorical feature to dummy variables
inputs = pd.get_dummies(inputs, columns=['mileage_tier'], drop_first=True)
# Add log-transformed features (add 1 to avoid log(0))
inputs['log_miles_traveled'] = np.log1p(inputs['miles_traveled'])
inputs['log_total_receipts_amount'] = np.log1p(inputs['total_receipts_amount'])
inputs = inputs.loc[:, ~inputs.columns.duplicated()]
print(list(inputs.columns))  # To verify

# Ensure all feature columns are numeric
inputs = inputs.astype(float)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Define refined hyperparameter grid
param_grid = {
    'n_estimators': [150, 200, 250],  # Moderate tree count
    'max_depth': [10, 15, 20],        # Controlled depth
    'min_samples_split': [5, 10]      # Prevent overfitting
}

# Initialize and tune Random Forest
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Predict and evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error on Test Set: ${mae:.2f}")

# Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': inputs.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:\n", feature_importance.head(10))  # Top 10 features

# Save the refined model
joblib.dump(best_model, '/content/reimbursement_model.pkl')
print("Model saved to /content/reimbursement_model.pkl")