import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Import feature definitions and pipeline helpers
try:
    from src.preprocessing import (
        impute_missing_values,
        add_stage2_engineered_features,
        get_feature_definitions
    )
except ModuleNotFoundError:
    from preprocessing import (
        impute_missing_values,
        add_stage2_engineered_features,
        get_feature_definitions
    )

FEATURE_NAME_MAPPING = {
    'Area_sqft': 'Building Area (sq.ft)',
    'Number_of_Floors': 'Number of Floors',
    'Material_Cost': 'Estimated Material Cost',
    'Labour_Cost': 'Estimated Labour Cost',
    'Labour_Count': 'Baseline Labour Count',
    'Equipment_Cost': 'Estimated Equipment Cost',
    'Planned_Duration': 'Planned Duration (Days)',
    'Baseline_Cost': 'Baseline Estimated Cost',
    'Planned_Progress_Percent': 'Planned Progress (%)',
    'Actual_Progress_Percent': 'Actual Progress (%)',
    'Actual_Labour_Count': 'Actual Labour Count',
    'Material_Availability_Percent': 'Material Availability (%)',
    'Equipment_Availability_Percent': 'Equipment Availability (%)',
    'Progress_Gap': 'Schedule Progress Gap (%)',
    'Labour_Deployment_Ratio': 'Labour Deployment Ratio',
    'Labour_Shortage': 'Labour Shortage (Workers)',
    'Labour_Deployment_Gap': 'Labour Deployment Gap',
    'Material_Shortage': 'Material Shortage (%)',
    'Equipment_Shortage': 'Equipment Shortage (%)',
    'Progress_Ratio': 'Progress Ratio',
    'Resource_Availability_Index': 'Resource Availability Index',
    'Resource_Stress_Index': 'Resource Stress Index',
    'Project_Size_Duration_Ratio': 'Project Size to Duration Ratio',
    'Area_Per_Floor': 'Area Per Floor (sq.ft)',
    'Work_Rate_Deficit': 'Work Rate Deficit',
    'Work_Deficit_Days': 'Work Deficit Days',
    'Planned_Work_Rate': 'Planned Work Rate',
    'Actual_Work_Rate': 'Actual Work Rate'
}

def get_readable_feature_name(encoded_name: str) -> str:
    """
    Translates raw/encoded feature names into clean, human-readable titles.
    """
    if encoded_name in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[encoded_name]
    
    for prefix in ['Project_Type_', 'Location_', 'Material_Type_', 'Equipment_Type_']:
        if encoded_name.startswith(prefix):
            category_title = prefix[:-1].replace('_', ' ')
            val_title = encoded_name[len(prefix):].replace('_', ' ')
            return f"{category_title}: {val_title}"
            
    return encoded_name.replace('_', ' ')

class ConstructionXAIExplainer:
    def __init__(self, models_dir="models"):
        """
        Loads saved trained models, preprocessing transformers, and sets up SHAP explainers.
        """
        self.stage1_model = joblib.load(os.path.join(models_dir, "stage1_cost_model.joblib"))
        self.stage2_delay_model = joblib.load(os.path.join(models_dir, "stage2_delay_model.joblib"))
        self.stage2_risk_model = joblib.load(os.path.join(models_dir, "stage2_risk_model.joblib"))
        
        self.stage1_ct = joblib.load(os.path.join(models_dir, "stage1_pipeline.joblib"))
        self.stage2_ct = joblib.load(os.path.join(models_dir, "stage2_pipeline.joblib"))
        self.le_risk = joblib.load(os.path.join(models_dir, "stage2_risk_label_encoder.joblib"))
        
        # Initialize SHAP TreeExplainers
        self.exp_stage1 = shap.TreeExplainer(self.stage1_model)
        self.exp_stage2_delay = shap.TreeExplainer(self.stage2_delay_model)
        self.exp_stage2_risk = shap.TreeExplainer(self.stage2_risk_model)
        
        # Extract encoded feature names
        stage1_cols, stage2_cols = get_feature_definitions()
        cat_cols = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']
        num_cols_s1 = [c for c in stage1_cols if c not in cat_cols]
        num_cols_s2 = [c for c in stage2_cols if c not in cat_cols]
        
        cat_s1_names = self.stage1_ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
        self.s1_feature_names = list(cat_s1_names) + num_cols_s1
        
        cat_s2_names = self.stage2_ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
        self.s2_feature_names = list(cat_s2_names) + num_cols_s2
        
        self.s1_readable_names = [get_readable_feature_name(f) for f in self.s1_feature_names]
        self.s2_readable_names = [get_readable_feature_name(f) for f in self.s2_feature_names]

    def preprocess_stage1_input(self, raw_input: pd.DataFrame) -> pd.DataFrame:
        clean = impute_missing_values(raw_input)
        stage1_cols, _ = get_feature_definitions()
        proc = clean[stage1_cols]
        transformed = self.stage1_ct.transform(proc)
        return pd.DataFrame(transformed, columns=self.s1_feature_names)

    def preprocess_stage2_input(self, raw_input: pd.DataFrame) -> pd.DataFrame:
        clean = impute_missing_values(raw_input)
        proc_eng = add_stage2_engineered_features(clean)
        _, stage2_cols = get_feature_definitions()
        proc = proc_eng[stage2_cols]
        transformed = self.stage2_ct.transform(proc)
        return pd.DataFrame(transformed, columns=self.s2_feature_names)

    def explain_stage1_cost(self, raw_input: pd.DataFrame, top_n: int = 5):
        """
        Explains Stage 1 Total Construction Cost prediction using TreeSHAP.
        """
        X_trans = self.preprocess_stage1_input(raw_input)
        
        pred_before = self.stage1_model.predict(X_trans)[0]
        pred_cost = max(0.0, float(pred_before))
        
        # Verify SHAP execution does not alter prediction
        shap_values_obj = self.exp_stage1(X_trans)
        pred_after = self.stage1_model.predict(X_trans)[0]
        assert np.isclose(pred_before, pred_after, atol=1e-5), "Prediction changed after SHAP!"
        
        shap_vals = shap_values_obj.values[0]
        base_val = shap_values_obj.base_values[0]
        
        shap_list = []
        for feat, read_feat, val in zip(self.s1_feature_names, self.s1_readable_names, shap_vals):
            direction = "increases" if val > 0 else "decreases"
            shap_list.append({
                "raw_feature": feat,
                "feature": read_feat,
                "shap_value": float(val),
                "direction": direction,
                "magnitude": abs(float(val))
            })
            
        shap_list.sort(key=lambda x: x["magnitude"], reverse=True)
        top_factors = shap_list[:top_n]
        
        # Generate human text
        text_lines = [f"Predicted Total Construction Cost: INR {pred_cost:,.2f}"]
        text_lines.append("Major Contributing Factors:")
        for idx, item in enumerate(top_factors, 1):
            text_lines.append(f"  {idx}. {item['feature']}: {item['direction'].capitalize()} predicted cost by INR {item['magnitude']:,.2f}")
            
        return {
            "prediction": pred_cost,
            "base_value": float(base_val),
            "top_factors": top_factors,
            "all_shap_values": shap_list,
            "human_explanation": "\n".join(text_lines)
        }

    def explain_stage2_delay(self, raw_input: pd.DataFrame, top_n: int = 5):
        """
        Explains Stage 2 Expected Delay Days prediction using TreeSHAP.
        """
        X_trans = self.preprocess_stage2_input(raw_input)
        
        pred_before = self.stage2_delay_model.predict(X_trans)[0]
        pred_delay = max(0.0, float(pred_before))
        
        shap_values_obj = self.exp_stage2_delay(X_trans)
        pred_after = self.stage2_delay_model.predict(X_trans)[0]
        assert np.isclose(pred_before, pred_after, atol=1e-5), "Prediction changed after SHAP!"
        
        shap_vals = shap_values_obj.values[0]
        base_val = shap_values_obj.base_values[0]
        
        shap_list = []
        for feat, read_feat, val in zip(self.s2_feature_names, self.s2_readable_names, shap_vals):
            direction = "increases" if val > 0 else "decreases"
            shap_list.append({
                "raw_feature": feat,
                "feature": read_feat,
                "shap_value": float(val),
                "direction": direction,
                "magnitude": abs(float(val))
            })
            
        shap_list.sort(key=lambda x: x["magnitude"], reverse=True)
        top_factors = shap_list[:top_n]
        
        text_lines = [f"Predicted Expected Delay: {pred_delay:.1f} days"]
        text_lines.append("Major Contributing Factors:")
        for idx, item in enumerate(top_factors, 1):
            text_lines.append(f"  {idx}. {item['feature']}: {item['direction'].capitalize()} predicted delay by {item['magnitude']:.1f} days")
            
        return {
            "prediction": pred_delay,
            "base_value": float(base_val),
            "top_factors": top_factors,
            "all_shap_values": shap_list,
            "human_explanation": "\n".join(text_lines)
        }

    def explain_stage2_risk(self, raw_input: pd.DataFrame, top_n: int = 5):
        """
        Explains Stage 2 Delay Risk Classification prediction using Multiclass TreeSHAP.
        Focuses SHAP explanation strictly on the actual predicted risk class.
        """
        X_trans = self.preprocess_stage2_input(raw_input)
        
        pred_encoded_before = self.stage2_risk_model.predict(X_trans)[0]
        pred_class = str(self.le_risk.inverse_transform([pred_encoded_before])[0])
        proba = self.stage2_risk_model.predict_proba(X_trans)[0]
        
        # Determine exact model classes directly without assuming class order
        model_classes = list(self.stage2_risk_model.classes_)
        class_idx = model_classes.index(pred_encoded_before)
        
        # Build probability mapping strictly aligned with model.classes_
        class_probs = {}
        for idx, c_val in enumerate(model_classes):
            label_name = str(self.le_risk.inverse_transform([c_val])[0])
            class_probs[label_name] = float(proba[idx])
        
        shap_values_obj = self.exp_stage2_risk(X_trans)
        pred_encoded_after = self.stage2_risk_model.predict(X_trans)[0]
        assert pred_encoded_before == pred_encoded_after, "Prediction changed after SHAP!"
        
        # Safely extract SHAP values and base_value for the predicted class
        # Handles Explanation object, 3D numpy array, and list of 2D arrays safely
        shap_vals = None
        base_val = 0.0
        n_features = len(self.s2_feature_names)
        n_classes = len(model_classes)
        
        if hasattr(shap_values_obj, 'values'):
            vals = shap_values_obj.values
            base_vals = shap_values_obj.base_values
            if isinstance(vals, np.ndarray) and vals.ndim == 3:
                # Shape could be (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features) or (n_classes, n_samples, n_features)
                if vals.shape[1] == n_features and vals.shape[2] == n_classes:
                    shap_vals = vals[0, :, class_idx]
                elif vals.shape[1] == n_classes and vals.shape[2] == n_features:
                    shap_vals = vals[0, class_idx, :]
                elif vals.shape[0] == n_classes and vals.shape[2] == n_features:
                    shap_vals = vals[class_idx, 0, :]
                else:
                    shap_vals = vals[0, :, class_idx]
                    
                if isinstance(base_vals, np.ndarray) and base_vals.ndim == 2:
                    base_val = float(base_vals[0, class_idx])
                elif isinstance(base_vals, (np.ndarray, list)):
                    base_val = float(base_vals[class_idx]) if len(base_vals) > class_idx else float(base_vals[0])
                else:
                    base_val = float(base_vals)
            elif isinstance(vals, list):
                shap_vals = vals[class_idx][0]
                base_val = float(base_vals[class_idx][0]) if isinstance(base_vals, list) else float(base_vals[0])
            else:
                shap_vals = vals[0]
                base_val = float(base_vals[0])
        elif isinstance(shap_values_obj, list):
            shap_vals = shap_values_obj[class_idx][0]
            if isinstance(self.exp_stage2_risk.expected_value, (list, np.ndarray)):
                base_val = float(self.exp_stage2_risk.expected_value[class_idx])
            else:
                base_val = float(self.exp_stage2_risk.expected_value)
        else:
            shap_vals = shap_values_obj[0]
            base_val = float(self.exp_stage2_risk.expected_value)
            
        assert len(shap_vals) == n_features, f"SHAP values feature dimension mismatch: expected {n_features}, got {len(shap_vals)}"
        
        shap_list = []
        for feat, read_feat, val in zip(self.s2_feature_names, self.s2_readable_names, shap_vals):
            is_pos = float(val) > 0
            direction = "supports_class" if is_pos else "moves_away"
            contrib_dir = f"Supports {pred_class} Risk prediction" if is_pos else f"Moves away from {pred_class} Risk prediction"
            shap_list.append({
                "raw_feature": feat,
                "feature": read_feat,
                "shap_value": float(val),
                "direction": direction,
                "contribution_direction": contrib_dir,
                "magnitude": abs(float(val))
            })
            
        shap_list.sort(key=lambda x: x["magnitude"], reverse=True)
        top_factors = shap_list[:top_n]
        
        # Build non-causal, academically rigorous human explanation
        text_lines = [f"Predicted Delay Risk: {pred_class} Risk (Probability: {class_probs[pred_class]*100:.1f}%)"]
        text_lines.append(f"SHAP explains how each feature contributed to the model's current prediction of {pred_class} Risk:")
        for idx, item in enumerate(top_factors, 1):
            text_lines.append(f"  {idx}. {item['feature']}: {item['contribution_direction']} (SHAP value: {item['shap_value']:+.4f})")
            
        return {
            "prediction": pred_class,
            "predicted_class": pred_class,
            "predicted_risk": pred_class,
            "risk_class_explained": pred_class,
            "probabilities": class_probs,
            "base_value": float(base_val),
            "top_factors": top_factors,
            "all_shap_values": shap_list,
            "human_explanation": "\n".join(text_lines)
        }

def generate_global_shap_visualizations(n_samples: int = 200, output_dir: str = "outputs/shap"):
    """
    Computes global SHAP feature importance and saves summary & bar plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = ConstructionXAIExplainer()
    
    df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
    sample_df = df_raw.sample(n=n_samples, random_state=42)
    
    X_s1_trans = explainer.preprocess_stage1_input(sample_df)
    X_s2_trans = explainer.preprocess_stage2_input(sample_df)
    
    # 1. Stage 1 Cost SHAP
    print("Generating Stage 1 Cost SHAP plots...")
    sv_s1 = explainer.exp_stage1(X_s1_trans)
    
    # Summary Dot Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_s1.values, X_s1_trans, feature_names=explainer.s1_readable_names, show=False)
    plt.title("Stage 1 Total Construction Cost — SHAP Summary Plot", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage1_cost_shap_summary.png"), dpi=300)
    plt.close()
    
    # Importance Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_s1.values, X_s1_trans, feature_names=explainer.s1_readable_names, plot_type="bar", show=False)
    plt.title("Stage 1 Total Construction Cost — Global Feature Importance", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage1_cost_shap_importance.png"), dpi=300)
    plt.close()
    
    # 2. Stage 2 Delay SHAP
    print("Generating Stage 2 Delay SHAP plots...")
    sv_s2_delay = explainer.exp_stage2_delay(X_s2_trans)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_s2_delay.values, X_s2_trans, feature_names=explainer.s2_readable_names, show=False)
    plt.title("Stage 2 Expected Delay Days — SHAP Summary Plot", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage2_delay_shap_summary.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv_s2_delay.values, X_s2_trans, feature_names=explainer.s2_readable_names, plot_type="bar", show=False)
    plt.title("Stage 2 Expected Delay Days — Global Feature Importance", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage2_delay_shap_importance.png"), dpi=300)
    plt.close()
    
    # 3. Stage 2 Risk SHAP (Aggregated across classes for global plot)
    print("Generating Stage 2 Risk SHAP plots...")
    sv_s2_risk = explainer.exp_stage2_risk(X_s2_trans)
    # sv_s2_risk.values shape is (200, 59, 3)
    # Mean absolute SHAP value across classes for global summary
    mean_abs_risk_shap = np.mean(np.abs(sv_s2_risk.values), axis=2)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(mean_abs_risk_shap, X_s2_trans, feature_names=explainer.s2_readable_names, show=False)
    plt.title("Stage 2 Delay Risk — SHAP Summary Plot", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage2_risk_shap_summary.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(mean_abs_risk_shap, X_s2_trans, feature_names=explainer.s2_readable_names, plot_type="bar", show=False)
    plt.title("Stage 2 Delay Risk — Global Feature Importance", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage2_risk_shap_importance.png"), dpi=300)
    plt.close()
    
    print("All 6 SHAP visualization plots created and saved successfully under outputs/shap/")

if __name__ == '__main__':
    explainer = ConstructionXAIExplainer()
    
    # Test local explanations on a sample record
    df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
    sample_row = df_raw.head(1)
    
    print("\n--- STAGE 1 LOCAL EXPLANATION TEST ---")
    s1_exp = explainer.explain_stage1_cost(sample_row)
    print(s1_exp['human_explanation'])
    
    print("\n--- STAGE 2 DELAY LOCAL EXPLANATION TEST ---")
    s2_del_exp = explainer.explain_stage2_delay(sample_row)
    print(s2_del_exp['human_explanation'])
    
    print("\n--- STAGE 2 RISK LOCAL EXPLANATION TEST ---")
    s2_risk_exp = explainer.explain_stage2_risk(sample_row)
    print(s2_risk_exp['human_explanation'])
    
    print("\n--- GENERATING GLOBAL SHAP PLOTS ---")
    generate_global_shap_visualizations(n_samples=200)
