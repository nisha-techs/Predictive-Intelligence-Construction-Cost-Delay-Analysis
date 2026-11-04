import pandas as pd
import numpy as np
import joblib
import shap
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.explainability import ConstructionXAIExplainer
from src.feature_engineering import add_stage2_engineered_features, impute_missing_values

explainer = ConstructionXAIExplainer()

le_risk = explainer.le_risk
print("LabelEncoder.classes_:", le_risk.classes_)
# In scikit-learn LabelEncoder with fit(['Low', 'Medium', 'High']):
# index 0: 'High'
# index 1: 'Low'
# index 2: 'Medium'
for i, c in enumerate(le_risk.classes_):
    print(f"Index {i}: {c}")

print("\nRandom Forest model classes_:", explainer.stage2_risk_model.classes_)

# Good project condition:
# Planned 50%, Actual 50% (Gap=0), Material 90%, Equipment 90%, Labour 50/50
good_df = pd.DataFrame([{
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
    'Actual_Progress_Percent': 50.0,
    'Actual_Labour_Count': 50,
    'Material_Availability_Percent': 90.0,
    'Equipment_Availability_Percent': 90.0
}])

# Poor project condition:
# Planned 50%, Actual 20% (Gap=30), Material 50%, Equipment 50%, Labour 25/50
poor_df = pd.DataFrame([{
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
    'Actual_Progress_Percent': 20.0,
    'Actual_Labour_Count': 25,
    'Material_Availability_Percent': 50.0,
    'Equipment_Availability_Percent': 50.0
}])

print("\n================ GOOD CASE INFERENCE ================")
X_good_trans = explainer.preprocess_stage2_input(good_df)
pred_good_enc = explainer.stage2_risk_model.predict(X_good_trans)[0]
proba_good = explainer.stage2_risk_model.predict_proba(X_good_trans)[0]
pred_good_str = le_risk.inverse_transform([pred_good_enc])[0]
print(f"Encoded Pred: {pred_good_enc} -> Class String: {pred_good_str}")
print("Probabilities per index:")
for i, c in enumerate(le_risk.classes_):
    print(f"  Index {i} ({c}): {proba_good[i]:.4f}")

shap_good_obj = explainer.exp_stage2_risk(X_good_trans)
print("SHAP base values shape/values:", shap_good_obj.base_values)
print("SHAP values shape:", shap_good_obj.values.shape)
# Shape is (1, n_features, n_classes)

exp_good_dict = explainer.explain_stage2_risk(good_df)
print("\nexplain_stage2_risk output for GOOD:")
print("Prediction:", exp_good_dict['prediction'])
print("Probabilities:", exp_good_dict['probabilities'])
print("Base value:", exp_good_dict['base_value'])
print("Top factors:")
for f in exp_good_dict['top_factors']:
    print(f"  Feature: {f['feature']:<30} | SHAP: {f['shap_value']:+.4f} | direction: {f['direction']}")

print("\n================ POOR CASE INFERENCE ================")
X_poor_trans = explainer.preprocess_stage2_input(poor_df)
pred_poor_enc = explainer.stage2_risk_model.predict(X_poor_trans)[0]
proba_poor = explainer.stage2_risk_model.predict_proba(X_poor_trans)[0]
pred_poor_str = le_risk.inverse_transform([pred_poor_enc])[0]
print(f"Encoded Pred: {pred_poor_enc} -> Class String: {pred_poor_str}")
print("Probabilities per index:")
for i, c in enumerate(le_risk.classes_):
    print(f"  Index {i} ({c}): {proba_poor[i]:.4f}")

exp_poor_dict = explainer.explain_stage2_risk(poor_df)
print("\nexplain_stage2_risk output for POOR:")
print("Prediction:", exp_poor_dict['prediction'])
print("Probabilities:", exp_poor_dict['probabilities'])
print("Base value:", exp_poor_dict['base_value'])
print("Top factors:")
for f in exp_poor_dict['top_factors']:
    print(f"  Feature: {f['feature']:<30} | SHAP: {f['shap_value']:+.4f} | direction: {f['direction']}")
