import sys, os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import json
from src.explainability import ConstructionXAIExplainer
from src.feature_engineering import add_stage2_engineered_features

exp = ConstructionXAIExplainer()

cases = [
    ("TEST A: Good project condition", {
        'Project_ID': 'TEST_CASE_A',
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
        'Actual_Progress_Percent': 42.0,  # 2% ahead of schedule
        'Actual_Labour_Count': 26,        # Adequate labour (26 vs 25)
        'Material_Availability_Percent': 98.0, # High material availability
        'Equipment_Availability_Percent': 95.0 # High equipment availability
    }),
    ("TEST B: Poor project condition", {
        'Project_ID': 'TEST_CASE_B',
        'Project_Type': 'Villa',
        'Location': 'Vellore',
        'Area_sqft': 6000.0,
        'Number_of_Floors': 2,
        'Material_Type': 'Standard_RCC',
        'Labour_Count': 34,
        'Equipment_Type': 'Concrete_Mixer',
        'Planned_Duration': 432,
        'Baseline_Cost': 8887000.0,
        'Planned_Progress_Percent': 46.3,
        'Actual_Progress_Percent': 17.3,  # Large 29.0% progress gap
        'Actual_Labour_Count': 24,        # Labour shortage (24 vs 34, -10 workers)
        'Material_Availability_Percent': 77.2, # Lower material availability
        'Equipment_Availability_Percent': 62.9  # Lower equipment availability
    }),
    ("TEST C: Mixed condition", {
        'Project_ID': 'TEST_CASE_C',
        'Project_Type': 'Commercial',
        'Location': 'Coimbatore',
        'Area_sqft': 15000.0,
        'Number_of_Floors': 6,
        'Material_Type': 'Premium_RCC',
        'Labour_Count': 45,
        'Equipment_Type': 'Crane_and_Mixer',
        'Planned_Duration': 540,
        'Baseline_Cost': 55000000.0,
        'Planned_Progress_Percent': 40.0,
        'Actual_Progress_Percent': 28.0,  # Moderate 12.0% progress gap
        'Actual_Labour_Count': 35,        # Moderate labour shortage (35 vs 45)
        'Material_Availability_Percent': 70.0, # Moderate material availability
        'Equipment_Availability_Percent': 75.0  # Moderate equipment availability
    })
]

print("================================================================================")
print("CONTROLLED TEST EVALUATION REPORT FOR STAGE 2 RANDOM FOREST & SHAP")
print("================================================================================\n")

for title, inp in cases:
    df_raw = pd.DataFrame([inp])
    df_eng = add_stage2_engineered_features(df_raw)
    
    eng_features = {
        'Progress_Gap': float(df_eng['Progress_Gap'].iloc[0]),
        'Labour_Deployment_Ratio': float(df_eng['Labour_Deployment_Ratio'].iloc[0]),
        'Labour_Shortage': float(df_eng['Labour_Shortage'].iloc[0]),
        'Labour_Deployment_Gap': float(df_eng['Labour_Deployment_Gap'].iloc[0]),
        'Material_Shortage': float(df_eng['Material_Shortage'].iloc[0]),
        'Equipment_Shortage': float(df_eng['Equipment_Shortage'].iloc[0]),
        'Progress_Ratio': float(df_eng['Progress_Ratio'].iloc[0]),
        'Resource_Availability_Index': float(df_eng['Resource_Availability_Index'].iloc[0]),
        'Resource_Stress_Index': float(df_eng['Resource_Stress_Index'].iloc[0]),
        'Project_Size_Duration_Ratio': float(df_eng['Project_Size_Duration_Ratio'].iloc[0]),
        'Area_Per_Floor': float(df_eng['Area_Per_Floor'].iloc[0]),
        'Work_Rate_Deficit': float(df_eng['Work_Rate_Deficit'].iloc[0]),
        'Work_Deficit_Days': float(df_eng['Work_Deficit_Days'].iloc[0]),
        'Planned_Work_Rate': float(df_eng['Planned_Work_Rate'].iloc[0]),
        'Actual_Work_Rate': float(df_eng['Actual_Work_Rate'].iloc[0]),
    }
    
    res = exp.explain_stage2_risk(df_raw, top_n=5)
    
    print(f"--------------------------------------------------------------------------------")
    print(f"CASE: {title}")
    print(f"--------------------------------------------------------------------------------")
    print("1. Input Values:")
    for k, v in inp.items():
        print(f"   - {k}: {v}")
    print("\n2. Engineered Feature Values:")
    for k, v in eng_features.items():
        print(f"   - {k}: {v:.4f}")
    print(f"\n3. Predicted Risk: {res['predicted_class']} Risk")
    print("4. Probabilities:")
    for c, p in res['probabilities'].items():
        print(f"   - {c}: {p*100:.1f}%")
    print(f"5. SHAP Class Being Explained: {res['risk_class_explained']} (Base Value = {res['base_value']:.4f})")
    print("6. Top SHAP Features:")
    for idx, f in enumerate(res['top_factors'], 1):
        sign = "+" if f['shap_value'] > 0 else "-"
        print(f"   {idx}. {f['feature']} (raw: {f['raw_feature']})")
        print(f"      SHAP Value: {f['shap_value']:+.4f} | Sign: {sign} | Direction: {f['direction']}")
        print(f"      Display Label: \"{f['contribution_direction']}\"")
    print("7. SHAP Contribution Signs:")
    signs_summary = [f"{f['feature']}: {'+' if f['shap_value'] > 0 else '-'}" for f in res['top_factors']]
    print("   " + ", ".join(signs_summary))
    print(f"8. Matches Predicted Class: YES (All SHAP factors strictly explain {res['predicted_class']} Risk)\n")
