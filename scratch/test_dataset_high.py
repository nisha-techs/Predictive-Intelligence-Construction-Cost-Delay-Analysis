import sys, os
sys.path.insert(0, ".")
import pandas as pd
from src.explainability import ConstructionXAIExplainer

exp = ConstructionXAIExplainer()
df = pd.read_csv('dataset/construction_cost_delay_dataset_clean.csv')

for idx in [5, 6, 18]:
    row = df.loc[idx].to_dict()
    row['Planned_Progress_Percent'] = row['Baseline_Cost.1']
    true_label = row['Delay_Risk']
    df_row = pd.DataFrame([row])
    
    X = exp.preprocess_stage2_input(df_row)
    pred_encoded = exp.stage2_risk_model.predict(X)[0]
    pred_class = exp.le_risk.inverse_transform([pred_encoded])[0]
    proba = exp.stage2_risk_model.predict_proba(X)[0]
    probs = {exp.le_risk.inverse_transform([c])[0]: float(proba[i]) for i, c in enumerate(exp.stage2_risk_model.classes_)}
    print(f"Row {idx} (True label: {true_label}) -> Predicted: {pred_class}, Probs: {probs}")
    exp_res = exp.explain_stage2_risk(df_row, top_n=3)
    print(f"  SHAP explains: {exp_res['predicted_class']}")
    for f in exp_res['top_factors']:
        print(f"    {f['feature']}: {f['shap_value']:+.4f} ({f['direction']})")
