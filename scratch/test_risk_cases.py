import sys, os
sys.path.insert(0, ".")
import pandas as pd
from src.explainability import ConstructionXAIExplainer

exp = ConstructionXAIExplainer()

cases = [
    ("Good (Ahead of schedule, 98% materials, 95% equipment, 100% labour)", {
        'Planned_Progress_Percent': 40.0, 'Actual_Progress_Percent': 45.0,
        'Actual_Labour_Count': 25, 'Labour_Count': 25,
        'Material_Availability_Percent': 98.0, 'Equipment_Availability_Percent': 95.0,
        'Planned_Duration': 360, 'Area_sqft': 5000.0, 'Number_of_Floors': 4,
        'Baseline_Cost': 15000000.0, 'Project_Type': 'Residential', 'Location': 'Chennai',
        'Material_Type': 'Standard_RCC', 'Equipment_Type': 'Concrete_Mixer'
    }),
    ("Medium (12% gap, 70% materials, 75% equipment, 72% labour)", {
        'Planned_Progress_Percent': 40.0, 'Actual_Progress_Percent': 28.0,
        'Actual_Labour_Count': 18, 'Labour_Count': 25,
        'Material_Availability_Percent': 70.0, 'Equipment_Availability_Percent': 75.0,
        'Planned_Duration': 360, 'Area_sqft': 5000.0, 'Number_of_Floors': 4,
        'Baseline_Cost': 15000000.0, 'Project_Type': 'Residential', 'Location': 'Chennai',
        'Material_Type': 'Standard_RCC', 'Equipment_Type': 'Concrete_Mixer'
    }),
    ("High (35% gap, 30% materials, 35% equipment, 30% labour)", {
        'Planned_Progress_Percent': 70.0, 'Actual_Progress_Percent': 25.0,
        'Actual_Labour_Count': 8, 'Labour_Count': 30,
        'Material_Availability_Percent': 30.0, 'Equipment_Availability_Percent': 35.0,
        'Planned_Duration': 360, 'Area_sqft': 8000.0, 'Number_of_Floors': 5,
        'Baseline_Cost': 25000000.0, 'Project_Type': 'Commercial', 'Location': 'Coimbatore',
        'Material_Type': 'Premium_RCC', 'Equipment_Type': 'Crane_and_Mixer'
    }),
    ("Severe High (50% gap, 15% materials, 20% equipment, 15% labour)", {
        'Planned_Progress_Percent': 80.0, 'Actual_Progress_Percent': 20.0,
        'Actual_Labour_Count': 5, 'Labour_Count': 40,
        'Material_Availability_Percent': 15.0, 'Equipment_Availability_Percent': 20.0,
        'Planned_Duration': 450, 'Area_sqft': 20000.0, 'Number_of_Floors': 8,
        'Baseline_Cost': 60000000.0, 'Project_Type': 'Hospital', 'Location': 'Madurai',
        'Material_Type': 'High_End_Finish', 'Equipment_Type': 'Heavy_Equipment_Set'
    })
]

for label, data in cases:
    df = pd.DataFrame([data])
    X = exp.preprocess_stage2_input(df)
    pred_encoded = exp.stage2_risk_model.predict(X)[0]
    pred_class = exp.le_risk.inverse_transform([pred_encoded])[0]
    proba = exp.stage2_risk_model.predict_proba(X)[0]
    probs = {exp.le_risk.inverse_transform([c])[0]: float(proba[i]) for i, c in enumerate(exp.stage2_risk_model.classes_)}
    print(f"\n=== {label} ===")
    print(f"  Prediction: {pred_class} Risk")
    print(f"  Probabilities: {probs}")
    exp_res = exp.explain_stage2_risk(df, top_n=5)
    print(f"  SHAP Explains Class: {exp_res['predicted_class']}")
    print(f"  Base Value: {exp_res['base_value']:.4f}")
    for tf in exp_res['top_factors']:
        print(f"    {tf['feature']}: {tf['shap_value']:+.4f} ({tf['direction']})")
