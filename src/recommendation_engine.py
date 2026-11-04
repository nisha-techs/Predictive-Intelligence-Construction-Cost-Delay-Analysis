import os
import pandas as pd
import numpy as np

# Import XAI explainer
try:
    from src.explainability import ConstructionXAIExplainer, get_readable_feature_name
except ModuleNotFoundError:
    from explainability import ConstructionXAIExplainer, get_readable_feature_name

class ConstructionRecommendationEngine:
    def __init__(self, models_dir="models"):
        """
        Initializes the recommendation engine connected to the XAI explainer.
        """
        self.explainer = ConstructionXAIExplainer(models_dir=models_dir)

    def generate_stage1_cost_recommendations(self, raw_input_row: pd.DataFrame, top_n: int = 4) -> dict:
        """
        Generates actionable pre-construction cost recommendations based on Stage 1 predictions and SHAP analysis.
        """
        explanation = self.explainer.explain_stage1_cost(raw_input_row, top_n=10)
        pred_cost = explanation["prediction"]
        top_factors = explanation["top_factors"]
        
        row_dict = raw_input_row.iloc[0].to_dict()
        area = row_dict.get('Area_sqft', 1.0)
        
        recommendations = []
        
        for item in top_factors:
            raw_feat = item["raw_feature"]
            feat_name = item["feature"]
            shap_val = item["shap_value"]
            direction = item["direction"]
            
            # Only focus on cost-increasing factors for corrective recommendations
            if direction != "increases":
                continue
                
            if raw_feat == "Material_Cost":
                mat_cost = row_dict.get("Material_Cost", 0)
                mat_per_sqft = mat_cost / area if area > 0 else 0
                priority = "HIGH" if shap_val > 5_000_000 else "MEDIUM"
                recommendations.append({
                    "factor": "Material Cost",
                    "priority": priority,
                    "issue": f"Estimated material cost (INR {mat_cost:,.2f}, approx INR {mat_per_sqft:,.2f}/sq.ft) significantly increases total cost.",
                    "action": "Review material procurement rates and evaluate available cost-effective alternatives without compromising required specifications.",
                    "shap_contribution": shap_val
                })
                
            elif raw_feat == "Labour_Cost":
                lab_cost = row_dict.get("Labour_Cost", 0)
                lab_per_sqft = lab_cost / area if area > 0 else 0
                priority = "HIGH" if shap_val > 5_000_000 else "MEDIUM"
                recommendations.append({
                    "factor": "Labour Cost",
                    "priority": priority,
                    "issue": f"Estimated labour cost (INR {lab_cost:,.2f}, approx INR {lab_per_sqft:,.2f}/sq.ft) adds a substantial cost premium.",
                    "action": "Review labour allocation, productivity and current labour rates before finalizing the project budget.",
                    "shap_contribution": shap_val
                })
                
            elif raw_feat == "Equipment_Cost":
                eq_cost = row_dict.get("Equipment_Cost", 0)
                priority = "MEDIUM"
                recommendations.append({
                    "factor": "Equipment Cost",
                    "priority": priority,
                    "issue": f"Equipment cost estimate (INR {eq_cost:,.2f}) contributes towards higher overall budget requirements.",
                    "action": "Review equipment utilization and consider whether rental, scheduling or alternative equipment arrangements can reduce unnecessary equipment expenditure.",
                    "shap_contribution": shap_val
                })
                
            elif raw_feat in ["Area_sqft", "Number_of_Floors"]:
                floors = row_dict.get("Number_of_Floors", 1)
                priority = "MEDIUM"
                recommendations.append({
                    "factor": "Building Scale (Area / Floors)",
                    "priority": priority,
                    "issue": f"Building scale ({area:,.0f} sq.ft across {floors} floors) is a major baseline cost driver.",
                    "action": "Optimize floor plan layout and structural efficiency to balance space requirements with total building budget.",
                    "shap_contribution": shap_val
                })

        # Sort recommendations by priority and magnitude
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        recommendations.sort(key=lambda x: (priority_map[x["priority"]], x["shap_contribution"]), reverse=True)
        final_recs = recommendations[:top_n]
        
        if not final_recs:
            final_recs.append({
                "factor": "Budget Monitoring",
                "priority": "LOW",
                "issue": "Pre-construction cost estimates align well with standard baseline parameters.",
                "action": "Continue monitoring material and labour market rates prior to procurement execution.",
                "shap_contribution": 0.0
            })
            
        # Build summary text
        summary_lines = [f"Predicted Total Construction Cost: INR {pred_cost:,.2f}"]
        summary_lines.append(f"Pre-Construction Decision Support Suggestions ({len(final_recs)} Actions):")
        for idx, rec in enumerate(final_recs, 1):
            summary_lines.append(f"  {idx}. [{rec['priority']}] {rec['factor']}: {rec['action']}")
            
        return {
            "prediction": pred_cost,
            "recommendations": final_recs,
            "summary_text": "\n".join(summary_lines)
        }

    def generate_stage2_ongoing_recommendations(self, raw_input_row: pd.DataFrame, top_n: int = 4) -> dict:
        """
        Generates actionable ongoing delay and risk recommendations based on Stage 2 predictions and SHAP analysis.
        """
        delay_exp = self.explainer.explain_stage2_delay(raw_input_row, top_n=10)
        risk_exp = self.explainer.explain_stage2_risk(raw_input_row, top_n=10)
        
        pred_delay = delay_exp["prediction"]
        pred_risk = risk_exp["prediction"]
        risk_probs = risk_exp["probabilities"]
        
        row_dict = raw_input_row.iloc[0].to_dict()
        proj_id = str(row_dict.get('Project_ID', 'N/A'))
        
        planned_prog = float(row_dict.get('Planned_Progress_Percent', 0))
        actual_prog = float(row_dict.get('Actual_Progress_Percent', 0))
        progress_gap = planned_prog - actual_prog
        
        mat_avail = float(row_dict.get('Material_Availability_Percent', 100))
        eq_avail = float(row_dict.get('Equipment_Availability_Percent', 100))
        
        actual_labour = float(row_dict.get('Actual_Labour_Count', 0))
        baseline_labour = float(row_dict.get('Labour_Count', 1))
        labour_ratio = actual_labour / baseline_labour if baseline_labour > 0 else 1.0
        
        risk_top_factors = risk_exp["top_factors"]
        delay_top_factors = delay_exp["top_factors"]
        
        # Combine SHAP insights for delay and risk
        risk_shap_dict = {f["raw_feature"]: f for f in risk_top_factors}
        delay_shap_dict = {f["raw_feature"]: f for f in delay_top_factors}
        
        recommendations = []
        
        # 1. Progress Gap Recommendation
        if progress_gap > 3.0:
            shap_info = risk_shap_dict.get('Progress_Gap') or delay_shap_dict.get('Progress_Gap')
            shap_val = shap_info['shap_value'] if shap_info else 0.5
            
            if progress_gap > 15.0 or pred_risk == 'HIGH':
                priority = 'HIGH'
            elif progress_gap > 7.0 or pred_risk == 'MEDIUM':
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
                
            recommendations.append({
                "factor": "Progress Gap",
                "priority": priority,
                "issue": f"Actual progress ({actual_prog:.1f}%) is {progress_gap:.1f}% behind planned progress ({planned_prog:.1f}%).",
                "action": "Review the current work schedule and identify activities responsible for the progress shortfall. Consider resource reallocation or schedule recovery actions.",
                "shap_contribution": abs(shap_val)
            })
            
        # 2. Material Availability Recommendation
        if mat_avail < 80.0:
            shap_info = risk_shap_dict.get('Material_Availability_Percent') or delay_shap_dict.get('Material_Availability_Percent')
            shap_val = shap_info['shap_value'] if shap_info else 0.4
            
            priority = 'HIGH' if mat_avail < 65.0 or pred_risk == 'HIGH' else 'MEDIUM'
            recommendations.append({
                "factor": "Material Availability",
                "priority": priority,
                "issue": f"Material availability ({mat_avail:.1f}%) is below optimal site requirement.",
                "action": "Prioritize procurement of critical materials and verify supplier delivery schedules to reduce material-related work interruptions.",
                "shap_contribution": abs(shap_val)
            })

        # 3. Equipment Availability Recommendation
        if eq_avail < 80.0:
            shap_info = risk_shap_dict.get('Equipment_Availability_Percent') or delay_shap_dict.get('Equipment_Availability_Percent')
            shap_val = shap_info['shap_value'] if shap_info else 0.4
            
            priority = 'HIGH' if eq_avail < 65.0 or pred_risk == 'HIGH' else 'MEDIUM'
            recommendations.append({
                "factor": "Equipment Availability",
                "priority": priority,
                "issue": f"Equipment availability ({eq_avail:.1f}%) is below operational threshold.",
                "action": "Review equipment availability and downtime. Consider alternate equipment arrangements or revised equipment scheduling for critical activities.",
                "shap_contribution": abs(shap_val)
            })

        # 4. Labour Deployment Recommendation
        if labour_ratio < 0.85:
            shap_info = risk_shap_dict.get('Labour_Deployment_Ratio') or delay_shap_dict.get('Labour_Deployment_Ratio')
            shap_val = shap_info['shap_value'] if shap_info else 0.3
            
            priority = 'HIGH' if labour_ratio < 0.70 or pred_risk == 'HIGH' else 'MEDIUM'
            recommendations.append({
                "factor": "Labour Deployment",
                "priority": priority,
                "issue": f"Actual labour deployed ({actual_labour:.0f}) is lower than baseline requirement ({baseline_labour:.0f}) (ratio: {labour_ratio:.2f}).",
                "action": "Review labour deployment for critical activities and consider reallocating or adding resources where required.",
                "shap_contribution": abs(shap_val)
            })

        # Sort recommendations by priority and SHAP magnitude
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        recommendations.sort(key=lambda x: (priority_map[x["priority"]], x["shap_contribution"]), reverse=True)
        final_recs = recommendations[:top_n]
        
        # Risk Priority Message
        risk_upper = str(pred_risk).upper()
        if risk_upper == 'HIGH':
            header_msg = "HIGH DELAY RISK DETECTED: Immediate review of the major contributing project factors is recommended."
        elif risk_upper == 'MEDIUM':
            header_msg = "MODERATE DELAY RISK DETECTED: Review the identified schedule and resource factors before they develop into major delays."
        else:
            header_msg = "LOW DELAY RISK DETECTED: Current indicators suggest relatively low delay risk. Continue monitoring progress, material availability and equipment availability."

        if not final_recs and pred_risk == 'LOW':
            final_recs.append({
                "factor": "Site Monitoring",
                "priority": "LOW",
                "issue": "All progress and resource availability metrics are performing within normal operational parameters.",
                "action": "Continue regular site monitoring of progress, material supply chains, and equipment utilization.",
                "shap_contribution": 0.0
            })
            
        summary_lines = [
            f"Project ID: {proj_id}",
            f"Predicted Delay: {pred_delay:.1f} days | Predicted Risk Class: {pred_risk} (Probability: {risk_probs[pred_risk]*100:.1f}%)",
            f"\n{header_msg}\n",
            f"Key Action Items ({len(final_recs)} Recommendations):"
        ]
        for idx, rec in enumerate(final_recs, 1):
            summary_lines.append(f"  {idx}. [{rec['priority']}] {rec['factor']}: {rec['action']}")
            
        return {
            "project_id": proj_id,
            "predictions": {
                "expected_delay_days": pred_delay,
                "delay_risk": pred_risk,
                "risk_probabilities": risk_probs
            },
            "header_message": header_msg,
            "recommendations": final_recs,
            "summary_text": "\n".join(summary_lines)
        }

def run_recommendation_engine_tests(output_csv: str = "outputs/recommendation_results.csv"):
    """
    Validates recommendation engine across required test scenarios:
    1. High delay-risk scenario
    2. Medium delay-risk scenario
    3. Low delay-risk scenario
    4. Low material availability
    5. Low equipment availability
    6. Progress behind schedule
    7. Normal/healthy project condition
    """
    print("==================================================")
    print("RUNNING RECOMMENDATION ENGINE VALIDATION SCENARIOS")
    print("==================================================")
    
    engine = ConstructionRecommendationEngine()
    df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
    
    # Select scenarios from real data matching specific conditions
    scenarios = []
    
    # Scenario 1: High risk
    high_risk_rows = df_raw[df_raw['Delay_Risk'] == 'High']
    if not high_risk_rows.empty:
        scenarios.append(('High Delay-Risk Scenario', high_risk_rows.iloc[0:1]))
        
    # Scenario 2: Medium risk
    med_risk_rows = df_raw[df_raw['Delay_Risk'] == 'Medium']
    if not med_risk_rows.empty:
        scenarios.append(('Medium Delay-Risk Scenario', med_risk_rows.iloc[0:1]))
        
    # Scenario 3: Low risk
    low_risk_rows = df_raw[df_raw['Delay_Risk'] == 'Low']
    if not low_risk_rows.empty:
        scenarios.append(('Low Delay-Risk Scenario', low_risk_rows.iloc[0:1]))
        
    # Scenario 4: Low material availability
    low_mat_rows = df_raw[df_raw['Material_Availability_Percent'] < 65.0]
    if not low_mat_rows.empty:
        scenarios.append(('Low Material Availability Scenario', low_mat_rows.iloc[0:1]))
        
    # Scenario 5: Low equipment availability
    low_eq_rows = df_raw[df_raw['Equipment_Availability_Percent'] < 65.0]
    if not low_eq_rows.empty:
        scenarios.append(('Low Equipment Availability Scenario', low_eq_rows.iloc[0:1]))
        
    # Scenario 6: Progress behind schedule (Gap > 15)
    df_raw['Gap'] = df_raw['Planned_Progress_Percent'] - df_raw['Actual_Progress_Percent']
    behind_rows = df_raw[df_raw['Gap'] > 15.0]
    if not behind_rows.empty:
        scenarios.append(('Progress Behind Schedule Scenario', behind_rows.iloc[0:1]))
        
    # Scenario 7: Healthy project (Low risk, Gap < 0, High availability)
    healthy_rows = df_raw[(df_raw['Delay_Risk'] == 'Low') & (df_raw['Gap'] <= 0) & (df_raw['Material_Availability_Percent'] >= 85.0)]
    if not healthy_rows.empty:
        scenarios.append(('Healthy Project Scenario', healthy_rows.iloc[0:1]))

    results_export = []
    
    for scenario_name, row in scenarios:
        print(f"\n---> Scenario: {scenario_name}")
        st1_res = engine.generate_stage1_cost_recommendations(row)
        st2_res = engine.generate_stage2_ongoing_recommendations(row)
        
        print(st1_res['summary_text'])
        print(st2_res['summary_text'])
        
        proj_id = st2_res['project_id']
        pred_cost = st1_res['prediction']
        pred_delay = st2_res['predictions']['expected_delay_days']
        pred_risk = st2_res['predictions']['delay_risk']
        
        for rec in st2_res['recommendations']:
            results_export.append({
                "Scenario": scenario_name,
                "Project_ID": proj_id,
                "Stage": "Stage 2 Ongoing",
                "Predicted_Delay_Days": pred_delay,
                "Predicted_Risk": pred_risk,
                "Factor": rec["factor"],
                "Priority": rec["priority"],
                "Issue": rec["issue"],
                "Action": rec["action"],
                "SHAP_Contribution": rec["shap_contribution"]
            })
            
    # Export results to CSV
    os.makedirs("outputs", exist_ok=True)
    df_export = pd.DataFrame(results_export)
    df_export.to_csv(output_csv, index=False)
    print(f"\nRecommendation test validation results saved to {output_csv}")
    
    return df_export

if __name__ == '__main__':
    run_recommendation_engine_tests()
