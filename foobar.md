Product Requirements Document (PRD)

1. Business Problem
ACME Corp relies on a 60-year-old legacy travel reimbursement system that processes employee travel expenses but lacks transparency and documentation. The system takes three inputs—trip duration (days), miles traveled, and total receipts amount—and outputs a single reimbursement amount. No one understands its logic, leading to inconsistent reimbursements, employee frustration, and operational inefficiencies. 8090 has developed a modern replacement system, but ACME Corp is confused by discrepancies between the legacy and new system outputs. To address this, 8090 must reverse-engineer the legacy system’s logic to replicate its behavior, including quirks, and explain why the new system is superior in terms of transparency, consistency, and fairness.
The challenge is to create a solution that:

Accurately replicates the legacy system’s outputs for 1,000 historical public test cases and 5,000 private cases.
Identifies key business logic and quirks (e.g., rounding errors, bonuses) to document the legacy system’s behavior.
Enables clear communication with ACME stakeholders to justify the new system’s improvements.

The development environment is constrained by a 2017 MacBook Pro with limited computational power, necessitating cloud-based tools for efficient model training.
2. Current Process
Employees submit travel data via a legacy interface, providing:

Trip Duration: Number of days spent traveling (integer).
Miles Traveled: Total miles driven (integer or float).
Total Receipts Amount: Sum of expense receipts (float).

The system returns a single reimbursement amount (float, rounded to two decimal places) without explanation. Historical data (1,000 public cases) and employee interviews reveal complex, inconsistent logic, including:

Base per diem (~$100/day).
Tiered mileage rates (e.g., ~58 cents/mile for short trips, diminishing for longer trips).
Penalties for low receipts (<$50).
Bonuses for specific trip lengths (e.g., 5 days) or efficiency (miles/day).
Potential bugs (e.g., extra reimbursement for receipts ending in 49 cents).

The lack of source code and formal documentation, combined with anecdotal and conflicting employee insights, makes manual reverse-engineering challenging. The 2017 MacBook Pro’s limited CPU and lack of CUDA-compatible GPU further complicate local model training.
3. Project Goal
Develop a machine learning-based solution to:

Replicate Legacy Behavior: Accurately predict reimbursement amounts for given inputs, matching historical outputs (including quirks) within ±$0.01 for exact matches and ±$1.00 for close matches.
Understand Business Logic: Identify key rules, thresholds, and quirks (e.g., bonuses, penalties, rounding errors) to document the legacy system’s behavior.
Enable Comparison: Provide insights to explain discrepancies between the legacy and new systems, justifying the new system’s improvements (e.g., transparency, fairness).
Optimize Development: Use cloud-based training to overcome hardware limitations of a 2017 MacBook Pro, ensuring efficient development and testing.

The solution will be evaluated against 1,000 public test cases using the provided eval.sh script and 5,000 private cases for final scoring. Success is defined by high exact matches, low average error, and actionable insights into the legacy logic.
4. Product Description
The solution is a Python-based ML system that uses a tree-based ensemble model (e.g., Random Forest or Gradient Boosting) to predict reimbursement amounts. It leverages feature engineering, cloud-based training, and interpretability tools to achieve high accuracy and extract business logic. The system will:

Accept Inputs: Process trip_duration_days, miles_traveled, and total_receipts_amount via a run.sh script.
Predict Outputs: Output a single reimbursement amount (float, rounded to two decimal places) per test case.
Identify Quirks: Use interpretability tools (e.g., SHAP) to uncover anomalies like rounding bugs or penalties.
Run Efficiently: Execute in under 5 seconds per test case, with training offloaded to cloud platforms (e.g., Google Colab).
Support Submission: Generate results for private cases using generate_results.sh and submit via GitHub.

The development workflow combines local coding on a 2017 MacBook Pro with cloud-based model training to ensure scalability and speed.
5. Requirements
5.1 Functional Requirements

Input Processing:
Accept three inputs: trip_duration_days (integer), miles_traveled (integer/float), total_receipts_amount (float).
Parse inputs via run.sh script, integrating with eval.sh and generate_results.sh.

Model Training:
Train a tree-based ensemble model (e.g., Random Forest, XGBoost) on 1,000 public test cases from public_cases.json.
Use feature engineering to include derived features (e.g., miles per day, daily receipts, receipt cents).
Perform hyperparameter tuning (e.g., grid search) to optimize accuracy.

Prediction:
Generate reimbursement predictions rounded to two decimal places (e.g., $364.51).
Ensure predictions run in <5 seconds per test case on a 2017 MacBook Pro.

Interpretability:
Use SHAP to analyze feature contributions and identify quirks (e.g., penalties for low receipts, bonuses for 5-day trips).
Extract rules or thresholds (e.g., “if receipts < $50, subtract $20”) for documentation.

Output Generation:
Produce private_results.txt with one reimbursement per line for private cases using generate_results.sh.

Submission:
Package solution in a GitHub repository with run.sh, model files, and documentation.
Add arjun-krishna1 as a collaborator and submit via Google Form.

5.2 Technical Requirements

Local Development (2017 MacBook Pro):
Python: Version 3.8–3.10, installed via Anaconda.
Libraries: scikit-learn (Random Forest), pandas, numpy, SHAP, joblib (model serialization).
Tools: Jupyter Notebook for data analysis, VS Code/PyCharm for coding, Git for version control.
Use: Data preprocessing, feature engineering, lightweight model testing, inference, and quirk analysis.

Cloud-Based Training:
Platform: Google Colab (free tier, GPU-enabled) for training and tuning.
Alternative Platforms: Kaggle Notebooks (free) or AWS SageMaker (free tier) if Colab limits are reached.
Libraries: scikit-learn, XGBoost, or LightGBM for faster training.
Use: Train models, perform grid search, export models (e.g., model.pkl) for local inference.

Dependencies:
Ensure no external dependencies (e.g., network calls, databases) for run.sh execution.
Install jq and bc for eval.sh and generate_results.sh (e.g., brew install jq bc).

Performance:
Training: Complete in <10 minutes per model on Colab GPU.
Inference: <5 seconds per test case on 2017 MacBook Pro.

Compatibility:
Ensure scripts (run.sh, eval.sh, generate_results.sh) run on macOS (2017 MBP, likely Ventura or older).
Model outputs must be numeric and match evaluation format.

5.3 Non-Functional Requirements

Accuracy:
Achieve >95% exact matches (±$0.01) on public cases, maximizing close matches (±$1.00).
Minimize average error (<$0.10) and score (lower is better, per eval.sh).

Interpretability:
Document at least 3–5 key quirks or rules (e.g., rounding bugs, bonuses) with evidence (e.g., SHAP values, case examples).

Usability:
Provide clear documentation in GitHub README for setup, training, and quirk findings.
Ensure run.sh is simple to execute (e.g., ./run.sh 5 250 150.75).

Scalability:
Handle 5,000 private cases efficiently using cloud-trained model.

Cost:
Use free cloud tiers (Colab, Kaggle) to minimize expenses. Monitor AWS/GCP usage if used.

6. Solution Approach
6.1 Overview
The solution uses a tree-based ensemble ML model (e.g., Random Forest or XGBoost) to predict reimbursements, trained on 1,000 public cases. Feature engineering incorporates interview insights, and cloud-based training (Google Colab) overcomes hardware limitations. Interpretability tools (SHAP) identify quirks, enabling documentation of legacy logic.
6.2 Workflow

Data Analysis:
Load public_cases.json using pandas.
Analyze patterns (e.g., reimbursement vs. trip duration) in Jupyter Notebook.

Feature Engineering:
Create features: miles per day (miles_traveled / trip_duration_days), daily receipts (total_receipts_amount / trip_duration_days), receipt cents ((total_receipts_amount % 1) * 100), is_5_days (binary).
Validate features against interviews (e.g., efficiency bonuses, per diem).

Model Training:
Use Google Colab (GPU) to train Random Forest or XGBoost.
Perform 5-fold cross-validation and grid search (e.g., n_estimators, max_depth).
Export model using joblib (model.pkl).

Inference:
Integrate model into run.sh for local inference on 2017 MacBook Pro.
Example:import joblib
import sys
model = joblib.load('model.pkl')
inputs = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), ...]  # Add engineered features
print(f"{model.predict[[inputs]](0):.2f}")

Quirk Identification:
Use SHAP to analyze feature contributions for high-error or anomalous cases.
Cross-reference with interviews (e.g., low receipt penalties).
Document quirks (e.g., “Receipts < $50 reduce reimbursement by $20”).

Evaluation:
Run eval.sh locally to test against public cases, targeting >95% exact matches.
Analyze errors to refine model or add post-processing rules (e.g., adjust for rounding bugs).

Submission:
Generate private_results.txt using generate_results.sh on private cases.
Push code to GitHub, add collaborator, submit via Google Form.

6.3 Why ML?

Accuracy: Tree-based models capture complex, non-linear logic (e.g., bonuses, penalties) better than manual rule-coding.
Efficiency: Cloud training reduces development time vs. local training on 2017 MBP.
Interpretability: SHAP and rule extraction provide insights into quirks, supporting the mission to explain legacy logic.
Scalability: Handles 5,000 private cases efficiently.

7. Success Criteria

Accuracy:

95% exact matches (±$0.01) on public cases.

98% close matches (±$1.00).

Average error <$0.10, low score per eval.sh.
High performance on private cases (evaluated externally).

Interpretability:
Document 3–5 quirks/rules with evidence (e.g., SHAP values, case examples).
Example: “5-day trips with 180–220 miles/day receive $100 bonus, confirmed by SHAP and Lisa’s interview.”

Business Impact:
Provide a report summarizing legacy logic (e.g., per diem, mileage tiers, quirks).
Explain new system advantages (e.g., “Eliminates low-receipt penalties, improving fairness”).

Operational:
Complete training in <10 minutes per model on Colab.
Inference in <5 seconds per case on 2017 MBP.
Submit solution by deadline via GitHub and Google Form.

8. Risks and Mitigations

Risk: Slow training on 2017 MBP.
Mitigation: Use Google Colab GPU for training, reserving MBP for coding/inference.

Risk: Overfitting to public cases.
Mitigation: Use cross-validation, regularize model (e.g., limit max_depth), test on diverse inputs.

Risk: Missing quirks due to black-box model.
Mitigation: Apply SHAP, extract rules, validate against interviews and public cases.

Risk: Cloud platform limitations (e.g., Colab runtime disconnects).
Mitigation: Use Kaggle or AWS SageMaker as backups, save models frequently.

Risk: Incompatibility with eval.sh or generate_results.sh.
Mitigation: Test scripts early, ensure jq and bc are installed, validate output format.

9. Timeline

Week 1: Data analysis, feature engineering, local prototyping in Jupyter.
Week 2: Cloud-based training (Colab), hyperparameter tuning, SHAP analysis.
Week 3: Quirk identification, run.sh integration, local testing with eval.sh.
Week 4: Generate private results, finalize documentation, submit via GitHub/Form.

10. Stakeholders

8090 Team: Develops and submits solution.
**ACME Corp (Finance,
