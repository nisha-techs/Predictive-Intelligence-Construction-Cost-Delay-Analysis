import os
import pandas as pd
import numpy as np
from datetime import datetime

# Import database and XAI modules
try:
    from src.database import init_db, save_project, get_project, save_monitoring_record, list_projects, get_monitoring_history, update_project_completion
    from src.recommendation_engine import ConstructionRecommendationEngine
except ModuleNotFoundError:
    from database import init_db, save_project, get_project, save_monitoring_record, list_projects, get_monitoring_history, update_project_completion
    from recommendation_engine import ConstructionRecommendationEngine

class PredictionService:
    def __init__(self, models_dir="models"):
        """
        Loads models and XAI explainers ONCE at startup.
        Initializes database tables.
        """
        print("Initializing Prediction Service and loading ML models...")
        init_db()
        self.rec_engine = ConstructionRecommendationEngine(models_dir=models_dir)
        self.explainer = self.rec_engine.explainer
        print("All ML models, preprocessing pipelines, and explainers loaded successfully!")

    def validate_stage1_input(self, data: dict) -> tuple[bool, str]:
        """
        Validates Stage 1 input parameters.
        """
        required = [
            'Project_Type', 'Location', 'Area_sqft', 'Number_of_Floors',
            'Material_Type', 'Material_Cost', 'Labour_Cost', 'Labour_Count',
            'Equipment_Type', 'Equipment_Cost', 'Planned_Duration'
        ]
        for field in required:
            if field not in data or data[field] is None:
                return False, f"Missing required field: {field}"
                
        try:
            if float(data['Area_sqft']) <= 0:
                return False, "Area_sqft must be greater than 0."
            if int(data['Number_of_Floors']) <= 0:
                return False, "Number_of_Floors must be greater than 0."
            if float(data['Material_Cost']) < 0:
                return False, "Material_Cost cannot be negative."
            if float(data['Labour_Cost']) < 0:
                return False, "Labour_Cost cannot be negative."
            if int(data['Labour_Count']) < 0:
                return False, "Labour_Count cannot be negative."
            if float(data['Equipment_Cost']) < 0:
                return False, "Equipment_Cost cannot be negative."
            if int(data['Planned_Duration']) <= 0:
                return False, "Planned_Duration must be greater than 0."
        except (ValueError, TypeError) as e:
            return False, f"Invalid numeric input data type: {str(e)}"
            
        return True, ""

    def validate_stage2_input(self, data: dict) -> tuple[bool, str]:
        """
        Validates Stage 2 ongoing monitoring input parameters.
        """
        if 'Project_ID' not in data or not str(data['Project_ID']).strip():
            return False, "Missing required field: Project_ID"
            
        required = [
            'Planned_Progress_Percent', 'Actual_Progress_Percent',
            'Actual_Labour_Count', 'Material_Availability_Percent',
            'Equipment_Availability_Percent'
        ]
        for field in required:
            if field not in data or data[field] is None:
                return False, f"Missing required field: {field}"
                
        try:
            p_prog = float(data['Planned_Progress_Percent'])
            a_prog = float(data['Actual_Progress_Percent'])
            m_avail = float(data['Material_Availability_Percent'])
            e_avail = float(data['Equipment_Availability_Percent'])
            a_labour = int(data['Actual_Labour_Count'])
            
            if not (0 <= p_prog <= 100):
                return False, "Planned_Progress_Percent must be between 0 and 100."
            if not (0 <= a_prog <= 100):
                return False, "Actual_Progress_Percent must be between 0 and 100."
            if not (0 <= m_avail <= 100):
                return False, "Material_Availability_Percent must be between 0 and 100."
            if not (0 <= e_avail <= 100):
                return False, "Equipment_Availability_Percent must be between 0 and 100."
            if a_labour < 0:
                return False, "Actual_Labour_Count cannot be negative."
        except (ValueError, TypeError) as e:
            return False, f"Invalid numeric input data type: {str(e)}"
            
        return True, ""

    def predict_stage1_cost(self, data: dict) -> dict:
        """
        Executes Stage 1 Pre-Construction Cost Estimation:
        Input Validation -> Pipeline Preprocessing -> ML Inference -> SHAP XAI -> Recommendation Engine -> SQLite Storage
        """
        is_valid, err_msg = self.validate_stage1_input(data)
        if not is_valid:
            raise ValueError(err_msg)
            
        # Convert input dictionary to 1-row pandas DataFrame
        df_input = pd.DataFrame([data])
        
        # 1. SHAP Explanation & Inference
        exp_res = self.explainer.explain_stage1_cost(df_input, top_n=5)
        
        # 2. Recommendation Engine
        rec_res = self.rec_engine.generate_stage1_cost_recommendations(df_input, top_n=4)
        
        # 3. Store project record in SQLite
        db_payload = {
            'project_id': data.get('Project_ID'),
            'project_type': data['Project_Type'],
            'location': data['Location'],
            'area_sqft': data['Area_sqft'],
            'number_of_floors': data['Number_of_Floors'],
            'material_type': data['Material_Type'],
            'material_cost': data['Material_Cost'],
            'labour_cost': data['Labour_Cost'],
            'labour_count': data['Labour_Count'],
            'equipment_type': data['Equipment_Type'],
            'equipment_cost': data['Equipment_Cost'],
            'planned_duration': data['Planned_Duration'],
            'predicted_cost': exp_res['prediction']
        }
        saved_project_id = save_project(db_payload)
        
        return {
            "success": True,
            "stage": "cost_estimation",
            "project_id": saved_project_id,
            "prediction": {
                "total_construction_cost": exp_res["prediction"],
                "base_value": exp_res["base_value"]
            },
            "explanation": {
                "top_factors": exp_res["top_factors"],
                "human_explanation": exp_res["human_explanation"]
            },
            "recommendations": rec_res["recommendations"],
            "summary_text": rec_res["summary_text"]
        }

    def predict_stage2_ongoing(self, data: dict) -> dict:
        """
        Executes Stage 2 Ongoing Construction Monitoring:
        Project Retrieval -> Specs Merging -> ML Inference -> SHAP XAI -> Recommendation Engine -> SQLite History Storage
        """
        is_valid, err_msg = self.validate_stage2_input(data)
        if not is_valid:
            raise ValueError(err_msg)
            
        project_id = str(data['Project_ID']).strip()
        stored_proj = get_project(project_id)
        if not stored_proj:
            raise KeyError(f"No project exists with Project_ID {project_id}.")
            
        # Combine static Stage 1 specs stored in DB + ongoing Stage 2 monitoring inputs
        combined_data = {
            'Project_ID': project_id,
            'Project_Type': stored_proj['project_type'],
            'Location': stored_proj['location'],
            'Area_sqft': stored_proj['area_sqft'],
            'Number_of_Floors': stored_proj['number_of_floors'],
            'Material_Type': stored_proj['material_type'],
            'Labour_Count': stored_proj['labour_count'],
            'Equipment_Type': stored_proj['equipment_type'],
            'Planned_Duration': stored_proj['planned_duration'],
            'Baseline_Cost': stored_proj['predicted_cost'],  # Use estimated baseline cost
            'Planned_Progress_Percent': data['Planned_Progress_Percent'],
            'Actual_Progress_Percent': data['Actual_Progress_Percent'],
            'Actual_Labour_Count': data['Actual_Labour_Count'],
            'Material_Availability_Percent': data['Material_Availability_Percent'],
            'Equipment_Availability_Percent': data['Equipment_Availability_Percent']
        }
        
        df_combined = pd.DataFrame([combined_data])
        
        # 1. SHAP Explanations & Inference
        delay_exp = self.explainer.explain_stage2_delay(df_combined, top_n=5)
        risk_exp = self.explainer.explain_stage2_risk(df_combined, top_n=5)
        
        # 2. Recommendation Engine
        rec_res = self.rec_engine.generate_stage2_ongoing_recommendations(df_combined, top_n=4)
        
        # Check if monitoring completion was explicitly marked
        is_comp = data.get('is_completed') or data.get('is_complete')
        should_complete = is_comp in (1, True, '1', 'true', 'True') or float(data.get('Actual_Progress_Percent', 0)) >= 100
        if should_complete:
            update_project_completion(project_id, 1)

        # 3. Store monitoring record in SQLite
        pred_delay = float(delay_exp['prediction'])
        pred_risk = str(risk_exp['prediction'])
        risk_probs = risk_exp['probabilities']
        risk_prob = float(risk_probs[pred_risk])
        
        mon_payload = {
            'project_id': project_id,
            'planned_progress_percent': data['Planned_Progress_Percent'],
            'actual_progress_percent': data['Actual_Progress_Percent'],
            'actual_labour_count': data['Actual_Labour_Count'],
            'material_availability_percent': data['Material_Availability_Percent'],
            'equipment_availability_percent': data['Equipment_Availability_Percent'],
            'expected_delay_days': pred_delay,
            'delay_risk': pred_risk,
            'risk_probability': risk_prob
        }
        monitoring_id = save_monitoring_record(mon_payload)
        
        return {
            "success": True,
            "stage": "ongoing_monitoring",
            "project_id": project_id,
            "monitoring_id": monitoring_id,
            "is_completed": 1 if should_complete or stored_proj.get('is_completed') == 1 else 0,
            "prediction": {
                "expected_delay_days": pred_delay,
                "delay_risk": pred_risk,
                "risk_probabilities": risk_probs
            },
            "explanation": {
                "delay_top_factors": delay_exp["top_factors"],
                "risk_top_factors": risk_exp["top_factors"],
                "risk_class_explained": pred_risk,
                "predicted_risk": pred_risk,
                "delay_explanation": delay_exp["human_explanation"],
                "risk_explanation": risk_exp["human_explanation"]
            },
            "header_message": rec_res["header_message"],
            "recommendations": rec_res["recommendations"],
            "summary_text": rec_res["summary_text"]
        }

    def get_model_performance_summary(self) -> dict:
        """
        Reads stored model evaluation CSV files for the frontend metrics page.
        Sanitizes NaN/Inf values into valid JSON values.
        """
        import math

        def sanitize_nans(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return ""
            elif isinstance(obj, dict):
                return {k: sanitize_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_nans(v) for v in obj]
            return obj

        outputs_dir = "outputs"
        res = {"success": True, "performance": {}}
        
        files_map = {
            "stage1_cost": "stage1_model_comparison.csv",
            "stage2_delay": "stage2_delay_model_comparison.csv",
            "stage2_risk": "stage2_risk_model_comparison.csv"
        }
        
        for key, fname in files_map.items():
            fpath = os.path.join(outputs_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                df = df.where(pd.notnull(df), None)
                records = df.to_dict(orient="records")
                res["performance"][key] = sanitize_nans(records)
            else:
                res["performance"][key] = []
                
        return res
