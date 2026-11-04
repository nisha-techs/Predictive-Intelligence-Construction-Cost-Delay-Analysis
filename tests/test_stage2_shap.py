import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.explainability import ConstructionXAIExplainer

class Stage2ShapTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.explainer = ConstructionXAIExplainer()

    def test_01_model_classes_order(self):
        """Verify model.classes_ and label encoder mapping"""
        model = self.explainer.stage2_risk_model
        le = self.explainer.le_risk
        
        # Verify model classes are integers 0, 1, 2
        self.assertEqual(list(model.classes_), [0, 1, 2])
        
        # Verify LabelEncoder classes alphabetically: ['High', 'Low', 'Medium']
        self.assertEqual(list(le.classes_), ['High', 'Low', 'Medium'])
        self.assertEqual(le.inverse_transform([0])[0], 'High')
        self.assertEqual(le.inverse_transform([1])[0], 'Low')
        self.assertEqual(le.inverse_transform([2])[0], 'Medium')

    def test_02_feature_alignment(self):
        """Verify pipeline features match model feature order exactly"""
        model = self.explainer.stage2_risk_model
        self.assertEqual(len(self.explainer.s2_feature_names), model.n_features_in_)
        self.assertEqual(list(self.explainer.s2_feature_names), list(model.feature_names_in_))

    def test_03_good_condition_low_risk(self):
        """Test A: Good project condition -> Low Risk prediction and SHAP explanation"""
        test_a = {
            'Project_ID': 'TEST_LOW_RISK',
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
            'Actual_Progress_Percent': 42.0,
            'Actual_Labour_Count': 26,
            'Material_Availability_Percent': 98.0,
            'Equipment_Availability_Percent': 95.0
        }
        df = pd.DataFrame([test_a])
        res = self.explainer.explain_stage2_risk(df, top_n=5)
        
        self.assertEqual(res['predicted_class'], 'Low')
        self.assertEqual(res['predicted_risk'], 'Low')
        self.assertEqual(res['risk_class_explained'], 'Low')
        self.assertGreaterEqual(res['probabilities']['Low'], 0.5)
        
        # Top SHAP factors should explain Low Risk
        self.assertEqual(len(res['top_factors']), 5)
        for factor in res['top_factors']:
            self.assertIn("Low Risk prediction", factor['contribution_direction'])
            if factor['shap_value'] > 0:
                self.assertEqual(factor['direction'], 'supports_class')
                self.assertEqual(factor['contribution_direction'], 'Supports Low Risk prediction')
            else:
                self.assertEqual(factor['direction'], 'moves_away')
                self.assertEqual(factor['contribution_direction'], 'Moves away from Low Risk prediction')

    def test_04_poor_condition_high_risk(self):
        """Test B: Poor project condition -> High Risk prediction and SHAP explanation"""
        test_high = {
            'Project_ID': 'TEST_HIGH_RISK',
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
            'Actual_Progress_Percent': 17.3,  # 29% gap
            'Actual_Labour_Count': 24,        # 10 worker shortage
            'Material_Availability_Percent': 77.2,
            'Equipment_Availability_Percent': 62.9
        }
        df = pd.DataFrame([test_high])
        res = self.explainer.explain_stage2_risk(df, top_n=5)
        
        self.assertEqual(res['predicted_class'], 'High')
        self.assertEqual(res['predicted_risk'], 'High')
        self.assertEqual(res['risk_class_explained'], 'High')
        self.assertGreaterEqual(res['probabilities']['High'], 0.5)
        
        # Top SHAP factors should explain High Risk
        self.assertEqual(len(res['top_factors']), 5)
        for factor in res['top_factors']:
            self.assertIn("High Risk prediction", factor['contribution_direction'])
            if factor['shap_value'] > 0:
                self.assertEqual(factor['direction'], 'supports_class')
                self.assertEqual(factor['contribution_direction'], 'Supports High Risk prediction')
            else:
                self.assertEqual(factor['direction'], 'moves_away')
                self.assertEqual(factor['contribution_direction'], 'Moves away from High Risk prediction')

    def test_05_moderate_condition_medium_risk(self):
        """Test C: Moderate project condition -> Medium Risk prediction and SHAP explanation"""
        test_med = {
            'Project_ID': 'TEST_MED_RISK',
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
            'Actual_Progress_Percent': 28.0,  # 12% gap
            'Actual_Labour_Count': 18,        # Moderate shortage
            'Material_Availability_Percent': 70.0,
            'Equipment_Availability_Percent': 75.0
        }
        df = pd.DataFrame([test_med])
        res = self.explainer.explain_stage2_risk(df, top_n=5)
        
        self.assertEqual(res['predicted_class'], 'Medium')
        self.assertEqual(res['predicted_risk'], 'Medium')
        self.assertEqual(res['risk_class_explained'], 'Medium')
        self.assertGreaterEqual(res['probabilities']['Medium'], 0.5)
        
        # Top SHAP factors should explain Medium Risk
        self.assertEqual(len(res['top_factors']), 5)
        for factor in res['top_factors']:
            self.assertIn("Medium Risk prediction", factor['contribution_direction'])
            if factor['shap_value'] > 0:
                self.assertEqual(factor['direction'], 'supports_class')
                self.assertEqual(factor['contribution_direction'], 'Supports Medium Risk prediction')
            else:
                self.assertEqual(factor['direction'], 'moves_away')
                self.assertEqual(factor['contribution_direction'], 'Moves away from Medium Risk prediction')

if __name__ == '__main__':
    unittest.main()
