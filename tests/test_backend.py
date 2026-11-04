import os
import unittest
import json
import pandas as pd
import numpy as np

# Override DB_PATH in src.database BEFORE importing app
import src.database as db_module
TEST_DB_PATH = os.path.join("database", "test_app_storage.db")
db_module.DB_PATH = TEST_DB_PATH

# Import Flask app
from app import app
from src.database import get_db_connection

class BackendAPITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_module.DB_PATH = TEST_DB_PATH
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except OSError:
                pass
        db_module.init_db(clean_non_reference=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except OSError:
                pass
        db_module.DB_PATH = os.path.join("database", "app_storage.db")

    def setUp(self):
        db_module.DB_PATH = TEST_DB_PATH
        self.app = app.test_client()
        self.app.testing = True

    def test_01_health_check(self):
        """Test GET /api/health"""
        res = self.app.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['models_loaded'])
        print("  [PASSED] Test 1: GET /api/health")

    def test_02_stage1_predict_valid(self):
        """Test POST /api/stage1/predict with valid input"""
        payload = {
            "Project_Type": "Residential",
            "Location": "Chennai",
            "Area_sqft": 5000.0,
            "Number_of_Floors": 4,
            "Material_Type": "Standard_RCC",
            "Material_Cost": 4000000.0,
            "Labour_Cost": 800000.0,
            "Labour_Count": 25,
            "Equipment_Type": "Concrete_Mixer",
            "Equipment_Cost": 150000.0,
            "Planned_Duration": 360
        }
        res = self.app.post('/api/stage1/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('project_id', data)
        self.assertGreater(data['prediction']['total_construction_cost'], 0)
        self.assertIn('explanation', data)
        self.assertIn('recommendations', data)
        print(f"  [PASSED] Test 2: POST /api/stage1/predict (Generated Project_ID: {data['project_id']}, Predicted Cost: INR {data['prediction']['total_construction_cost']:,.2f})")
        return data['project_id']

    def test_03_stage1_predict_invalid(self):
        """Test POST /api/stage1/predict with invalid data (Area_sqft <= 0)"""
        payload = {
            "Project_Type": "Residential",
            "Location": "Chennai",
            "Area_sqft": -500.0,  # Invalid
            "Number_of_Floors": 4,
            "Material_Type": "Standard_RCC",
            "Material_Cost": 4000000.0,
            "Labour_Cost": 800000.0,
            "Labour_Count": 25,
            "Equipment_Type": "Concrete_Mixer",
            "Equipment_Cost": 150000.0,
            "Planned_Duration": 360
        }
        res = self.app.post('/api/stage1/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], 'Invalid input')
        print("  [PASSED] Test 3: POST /api/stage1/predict invalid area rejection")

    def test_04_get_projects(self):
        """Test GET /api/projects"""
        res = self.app.get('/api/projects')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['projects'], list)
        print(f"  [PASSED] Test 4: GET /api/projects (Found {data['count']} stored projects)")

    def test_05_get_single_project(self):
        """Test GET /api/projects/<project_id>"""
        # First create a project
        proj_id = self.test_02_stage1_predict_valid()
        res = self.app.get(f'/api/projects/{proj_id}')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['project']['project_id'], proj_id)
        print(f"  [PASSED] Test 5: GET /api/projects/{proj_id}")

    def test_06_stage2_predict_valid(self):
        """Test POST /api/stage2/predict with valid project and ongoing monitoring data"""
        proj_id = self.test_02_stage1_predict_valid()
        payload = {
            "Project_ID": proj_id,
            "Planned_Progress_Percent": 40.0,
            "Actual_Progress_Percent": 28.0,  # 12% gap
            "Actual_Labour_Count": 18,        # Low labour
            "Material_Availability_Percent": 70.0,
            "Equipment_Availability_Percent": 75.0
        }
        res = self.app.post('/api/stage2/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['project_id'], proj_id)
        self.assertGreaterEqual(data['prediction']['expected_delay_days'], 0)
        self.assertIn(data['prediction']['delay_risk'], ['Low', 'Medium', 'High'])
        self.assertIn('explanation', data)
        self.assertIn('recommendations', data)
        print(f"  [PASSED] Test 6: POST /api/stage2/predict (Predicted Delay: {data['prediction']['expected_delay_days']:.1f} days, Risk: {data['prediction']['delay_risk']})")

    def test_07_stage2_predict_invalid_percentages(self):
        """Test POST /api/stage2/predict with invalid percentage (> 100)"""
        proj_id = self.test_02_stage1_predict_valid()
        payload = {
            "Project_ID": proj_id,
            "Planned_Progress_Percent": 150.0,  # Invalid
            "Actual_Progress_Percent": 28.0,
            "Actual_Labour_Count": 18,
            "Material_Availability_Percent": 70.0,
            "Equipment_Availability_Percent": 75.0
        }
        res = self.app.post('/api/stage2/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], 'Invalid input')
        print("  [PASSED] Test 7: POST /api/stage2/predict invalid percentage rejection")

    def test_08_get_monitoring_history(self):
        """Test GET /api/projects/<project_id>/monitoring"""
        proj_id = self.test_02_stage1_predict_valid()
        # Create 2 monitoring snapshots
        payload1 = {
            "Project_ID": proj_id,
            "Planned_Progress_Percent": 20.0,
            "Actual_Progress_Percent": 18.0,
            "Actual_Labour_Count": 24,
            "Material_Availability_Percent": 90.0,
            "Equipment_Availability_Percent": 90.0
        }
        payload2 = {
            "Project_ID": proj_id,
            "Planned_Progress_Percent": 40.0,
            "Actual_Progress_Percent": 30.0,
            "Actual_Labour_Count": 18,
            "Material_Availability_Percent": 70.0,
            "Equipment_Availability_Percent": 70.0
        }
        self.app.post('/api/stage2/predict', data=json.dumps(payload1), content_type='application/json')
        self.app.post('/api/stage2/predict', data=json.dumps(payload2), content_type='application/json')
        
        res = self.app.get(f'/api/projects/{proj_id}/monitoring')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['count'], 2)
        self.assertEqual(data['monitoring_history'][0]['sequence_number'], 1)
        self.assertEqual(data['monitoring_history'][1]['sequence_number'], 2)
        print(f"  [PASSED] Test 8: GET /api/projects/{proj_id}/monitoring (Found {data['count']} snapshots)")

    def test_09_model_performance(self):
        """Test GET /api/model-performance"""
        res = self.app.get('/api/model-performance')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertIn('stage1_cost', data['performance'])
        self.assertIn('stage2_delay', data['performance'])
        self.assertIn('stage2_risk', data['performance'])
        print("  [PASSED] Test 9: GET /api/model-performance")

    def test_10_unknown_project_handling(self):
        """Test handling unknown project_id"""
        res = self.app.get('/api/projects/PRJ_UNKNOWN_9999')
        self.assertEqual(res.status_code, 404)
        data = json.loads(res.data)
        self.assertFalse(data['success'])
        self.assertEqual(data['error'], 'Project not found')
        print("  [PASSED] Test 10: Unknown Project_ID 404 handling")

    def test_11_full_project_lifecycle(self):
        """
        Complete End-to-End Project Lifecycle Test:
        Create Project (Stage 1) -> Retrieve Project -> Submit Ongoing Monitoring (Stage 2) -> Verify SHAP & Recommendations -> Get Monitoring History
        """
        print("\n==================================================")
        print("RUNNING END-TO-END PROJECT LIFECYCLE TEST")
        # Clean up any leftover test lifecycle records
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM monitoring_records WHERE project_id = 'PRJ_LIFECYCLE_01'")
        cursor.execute("DELETE FROM projects WHERE project_id = 'PRJ_LIFECYCLE_01'")
        conn.commit()
        conn.close()

        # 1. Submit Stage 1 Project Creation & Cost Estimation
        s1_payload = {
            "Project_ID": "PRJ_LIFECYCLE_01",
            "Project_Type": "Commercial",
            "Location": "Coimbatore",
            "Area_sqft": 15000.0,
            "Number_of_Floors": 6,
            "Material_Type": "Premium_RCC",
            "Material_Cost": 12000000.0,
            "Labour_Cost": 2500000.0,
            "Labour_Count": 45,
            "Equipment_Type": "Crane_and_Mixer",
            "Equipment_Cost": 400000.0,
            "Planned_Duration": 540
        }
        res1 = self.app.post('/api/stage1/predict', data=json.dumps(s1_payload), content_type='application/json')
        self.assertEqual(res1.status_code, 200)
        d1 = json.loads(res1.data)
        pid = d1['project_id']
        self.assertEqual(pid, "PRJ_LIFECYCLE_01")
        pred_cost = d1['prediction']['total_construction_cost']
        print(f"  Step 1: Created Project {pid} | Predicted Cost: INR {pred_cost:,.2f}")
        
        # 2. Retrieve Project Details
        res2 = self.app.get(f'/api/projects/{pid}')
        self.assertEqual(res2.status_code, 200)
        d2 = json.loads(res2.data)
        self.assertEqual(d2['project']['project_type'], "Commercial")
        print(f"  Step 2: Retrieved Project {pid} details successfully")
        
        # 3. Submit Stage 2 Ongoing Monitoring Snapshot 1
        s2_payload1 = {
            "Project_ID": pid,
            "Planned_Progress_Percent": 25.0,
            "Actual_Progress_Percent": 23.0,
            "Actual_Labour_Count": 42,
            "Material_Availability_Percent": 95.0,
            "Equipment_Availability_Percent": 90.0
        }
        res3 = self.app.post('/api/stage2/predict', data=json.dumps(s2_payload1), content_type='application/json')
        self.assertEqual(res3.status_code, 200)
        d3 = json.loads(res3.data)
        print(f"  Step 3: Snapshot 1 -> Delay: {d3['prediction']['expected_delay_days']:.1f} days, Risk: {d3['prediction']['delay_risk']}")
        
        # 4. Submit Stage 2 Ongoing Monitoring Snapshot 2 (Site Issue Introduced: 15% Gap, Low Materials)
        s2_payload2 = {
            "Project_ID": pid,
            "Planned_Progress_Percent": 50.0,
            "Actual_Progress_Percent": 35.0,
            "Actual_Labour_Count": 30,
            "Material_Availability_Percent": 60.0,
            "Equipment_Availability_Percent": 65.0
        }
        res4 = self.app.post('/api/stage2/predict', data=json.dumps(s2_payload2), content_type='application/json')
        self.assertEqual(res4.status_code, 200)
        d4 = json.loads(res4.data)
        print(f"  Step 4: Snapshot 2 -> Delay: {d4['prediction']['expected_delay_days']:.1f} days, Risk: {d4['prediction']['delay_risk']}")
        self.assertGreater(len(d4['recommendations']), 0)
        
        # 5. Retrieve Monitoring History
        res5 = self.app.get(f'/api/projects/{pid}/monitoring')
        self.assertEqual(res5.status_code, 200)
        d5 = json.loads(res5.data)
        self.assertEqual(d5['count'], 2)
        print(f"  Step 5: Retrieved {d5['count']} monitoring history records for {pid}")
        print("Full End-to-End Project Lifecycle Test Completed Successfully!")

    def test_12_project_completion_lifecycle(self):
        """
        Test completion lifecycle transition:
        Verify is_completed field in project details and /api/projects/<project_id>/complete endpoint.
        """
        proj_id = self.test_02_stage1_predict_valid()
        
        # Initial state check (is_completed should be 0)
        res1 = self.app.get(f'/api/projects/{proj_id}')
        d1 = json.loads(res1.data)
        self.assertEqual(d1['project'].get('is_completed', 0), 0)

        # Complete monitoring via POST /api/projects/<project_id>/complete
        res2 = self.app.post(f'/api/projects/{proj_id}/complete', data=json.dumps({"is_completed": 1}), content_type='application/json')
        self.assertEqual(res2.status_code, 200)
        d2 = json.loads(res2.data)
        self.assertTrue(d2['success'])
        self.assertEqual(d2['project']['is_completed'], 1)

        # Verify persistence via GET /api/projects/<project_id>
        res3 = self.app.get(f'/api/projects/{proj_id}')
        d3 = json.loads(res3.data)
        self.assertEqual(d3['project']['is_completed'], 1)
        print(f"  [PASSED] Test 12: Complete Project Monitoring Lifecycle for {proj_id}")

if __name__ == '__main__':
    unittest.main()
