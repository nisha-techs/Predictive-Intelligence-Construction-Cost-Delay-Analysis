import pandas as pd
import numpy as np
import joblib
import shap
import sys
import os

# Add root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.feature_engineering import add_stage2_engineered_features, impute_missing_values

print("=== 1. DATASET INVESTIGATION ===")
df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
print("Dataset shape:", df_raw.shape)
print("Columns:", df_raw.columns.tolist())

if 'Delay_Risk' in df_raw.columns:
    print("\nDelay_Risk distribution:")
    print(df_raw['Delay_Risk'].value_counts())

df_proc = add_stage2_engineered_features(impute_missing_values(df_raw))

print("\n--- Correlation of key features with Expected_Delay_Days ---")
cols_to_check = [
    'Expected_Delay_Days', 'Progress_Gap', 'Progress_Ratio', 'Material_Availability_Percent',
    'Equipment_Availability_Percent', 'Labour_Deployment_Ratio', 'Resource_Stress_Index',
    'Work_Rate_Deficit', 'Work_Deficit_Days', 'Actual_Work_Rate'
]
corr = df_proc[cols_to_check].corr()
print(corr['Expected_Delay_Days'])

print("\n--- Feature means grouped by Delay_Risk ---")
print(df_proc.groupby('Delay_Risk')[cols_to_check].mean())

print("\n=== 2. MODEL AND ENCODER ARTIFACT INVESTIGATION ===")
le_risk = joblib.load("models/stage2_risk_label_encoder.joblib")
print("LabelEncoder classes_:", le_risk.classes_)
for idx, cls in enumerate(le_risk.classes_):
    print(f"  Index {idx} -> Class '{cls}'")

model_risk = joblib.load("models/stage2_risk_model.joblib")
print("\nRandomForestClassifier classes_:", model_risk.classes_)
print("Number of estimators:", model_risk.n_estimators)

pipeline_s2 = joblib.load("models/stage2_pipeline.joblib")

print("\n=== 3. SHAP & EXPLAINABILITY INVESTIGATION ===")
from src.explainability import ConstructionXAIExplainer

explainer = ConstructionXAIExplainer()

# Controlled Test Case 1: Good Project Condition
good_case = pd.DataFrame([{
    'Project_ID': 'TEST_GOOD',
    'Project_Type': 'Commercial',
    'Location': 'Urban',
    'Area_sqft': 50000,
    'Number_of_Floors': 5,
    'Material_Type': 'Steel',
    'Labour_Count': 50,
    'Equipment_Type': 'Heavy',
    'Planned_Duration': 365,
    'Baseline_Cost': 50000000,
    'Planned_Progress_Percent': 50.0,
    'Actual_Progress_Percent': 55.0, # Ahead of schedule! Progress_Gap = -5
    'Actual_Labour_Count': 50,
    'Material_Availability_Percent': 95.0,
    'Equipment_Availability_Percent': 95.0
}])

# Controlled Test Case 2: Poor Project Condition
poor_case = pd.DataFrame([{
    'Project_ID': 'TEST_POOR',
    'Project_Type': 'Commercial',
    'Location': 'Urban',
    'Area_sqft': 50000,
    'Number_of_Floors': 5,
    'Material_Type': 'Steel',
    'Labour_Count': 50,
    'Equipment_Type': 'Heavy',
    'Planned_Duration': 365,
    'Baseline_Cost': 50000000,
    'Planned_Progress_Percent': 50.0,
    'Actual_Progress_Percent': 30.0, # Behind schedule! Progress_Gap = +20
    'Actual_Labour_Count': 25, # Labour shortage!
    'Material_Availability_Percent': 50.0, # Low material avail
    'Equipment_Availability_Percent': 50.0  # Low equipment avail
}])

print("\n--- EXPLAINING GOOD CASE ---")
res_good = explainer.explain_stage2_risk(good_case)
print("Prediction:", res_good['prediction'])
print("Probabilities:", res_good['probabilities'])
print("Top Factors:")
for f in res_good['top_factors']:
    print(f"  {f['feature']}: raw_val SHAP={f['shap_value']:.4f}, direction={f['direction']}")

print("\n--- EXPLAINING POOR CASE ---")
res_poor = explainer.explain_stage2_risk(poor_case)
print("Prediction:", res_poor['prediction'])
print("Probabilities:", res_poor['probabilities'])
print("Top Factors:")
for f in res_poor['top_factors']:
    print(f"  {f['feature']}: raw_val SHAP={f['shap_value']:.4f}, direction={f['direction']}")
