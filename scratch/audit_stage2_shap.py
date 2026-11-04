import os
import sys
sys.path.insert(0, os.path.abspath("."))
import joblib
import numpy as np
import pandas as pd
import shap

print("--- 1. ARTIFACTS VERIFICATION ---")
model = joblib.load("models/stage2_risk_model.joblib")
le = joblib.load("models/stage2_risk_label_encoder.joblib")
pipe = joblib.load("models/stage2_pipeline.joblib")

print("Model type:", type(model))
print("Model classes_:", model.classes_)
print("Model classes_ type:", type(model.classes_[0]))
print("LabelEncoder classes_:", le.classes_)
for idx, c in enumerate(model.classes_):
    label = le.inverse_transform([c])[0]
    print(f"  Class index {idx} in model.classes_ is {c} -> Label: '{label}'")

print("\n--- 2. PIPELINE & FEATURE ORDER ---")
cat_names = pipe.named_transformers_['cat'].get_feature_names_out(['Project_Type', 'Location', 'Material_Type', 'Equipment_Type'])
num_cols = pipe.transformers_[1][2]
pipe_feature_names = list(cat_names) + list(num_cols)
print(f"Pipeline feature count: {len(pipe_feature_names)}")

model_feature_names = list(model.feature_names_in_)
print(f"Model feature count: {len(model_feature_names)}")
print("Does pipeline feature order EXACTLY match model.feature_names_in_?", pipe_feature_names == model_feature_names)

print("\n--- 3. EXPLAINABILITY.PY FEATURE NAMES ---")
from src.explainability import ConstructionXAIExplainer
explainer = ConstructionXAIExplainer()
print("Explainer s2_feature_names == model.feature_names_in_?", explainer.s2_feature_names == model_feature_names)

print("\n--- 4. TEST CONTROLLED SAMPLES ---")
test_a = {
    'Project_ID': 'TEST_A',
    'Project_Type': 'Residential',
    'Location': 'Chennai',
    'Area_sqft': 5000.0,
    'Number_of_Floors': 4,
    'Material_Type': 'Standard_RCC',
    'Labour_Count': 25,
    'Equipment_Type': 'Concrete_Mixer',
    'Planned_Duration': 360,
    'Baseline_Cost': 15000000.0,
    'Planned_Progress_Percent': 40.0,
    'Actual_Progress_Percent': 42.0,  # Ahead of schedule
    'Actual_Labour_Count': 26,        # High labour
    'Material_Availability_Percent': 98.0, # High material
    'Equipment_Availability_Percent': 95.0 # High equipment
}

test_b = {
    'Project_ID': 'TEST_B',
    'Project_Type': 'Commercial',
    'Location': 'Coimbatore',
    'Area_sqft': 15000.0,
    'Number_of_Floors': 6,
    'Material_Type': 'Premium_RCC',
    'Labour_Count': 45,
    'Equipment_Type': 'Crane_and_Mixer',
    'Planned_Duration': 540,
    'Baseline_Cost': 55000000.0,
    'Planned_Progress_Percent': 60.0,
    'Actual_Progress_Percent': 30.0,  # 30% gap
    'Actual_Labour_Count': 20,        # Severe labour shortage (20 vs 45)
    'Material_Availability_Percent': 45.0, # Severe material shortage
    'Equipment_Availability_Percent': 50.0  # Severe equipment shortage
}

test_c = {
    'Project_ID': 'TEST_C',
    'Project_Type': 'Apartment',
    'Location': 'Madurai',
    'Area_sqft': 8000.0,
    'Number_of_Floors': 5,
    'Material_Type': 'Standard_RCC',
    'Labour_Count': 30,
    'Equipment_Type': 'Concrete_Mixer',
    'Planned_Duration': 400,
    'Baseline_Cost': 25000000.0,
    'Planned_Progress_Percent': 45.0,
    'Actual_Progress_Percent': 38.0,  # Moderate 7% gap
    'Actual_Labour_Count': 30,        # Adequate labour
    'Material_Availability_Percent': 65.0, # Moderate material shortage
    'Equipment_Availability_Percent': 85.0
}

for name, sample in [("TEST A (Good)", test_a), ("TEST B (Poor)", test_b), ("TEST C (Mixed)", test_c)]:
    df = pd.DataFrame([sample])
    
    # Preprocess
    X_trans = explainer.preprocess_stage2_input(df)
    
    pred_encoded = explainer.stage2_risk_model.predict(X_trans)[0]
    pred_class = explainer.le_risk.inverse_transform([pred_encoded])[0]
    proba = explainer.stage2_risk_model.predict_proba(X_trans)[0]
    
    print(f"\n=================== {name} ===================")
    print(f"Predicted encoded: {pred_encoded} -> Class: {pred_class}")
    print(f"Model classes: {explainer.stage2_risk_model.classes_}")
    print(f"LE classes: {explainer.le_risk.classes_}")
    for i, c in enumerate(explainer.stage2_risk_model.classes_):
        lbl = explainer.le_risk.inverse_transform([c])[0]
        print(f"  Prob of {lbl} (model class {c}, col {i}): {proba[i]:.4f}")
    
    # Check SHAP
    shap_obj = explainer.exp_stage2_risk(X_trans)
    print(f"SHAP values shape: {shap_obj.values.shape}")
    print(f"SHAP base_values: {shap_obj.base_values}")
    
    res = explainer.explain_stage2_risk(df)
    print(f"Explainer returned predicted_class: {res['predicted_class']}")
    print(f"Explainer base value: {res['base_value']:.4f}")
    print("Top factors in explain_stage2_risk:")
    for f in res['top_factors']:
        print(f"  {f['feature']} (raw: {f['raw_feature']}): shap={f['shap_value']:+.4f}, mag={f['magnitude']:.4f}, dir={f['direction']}")
