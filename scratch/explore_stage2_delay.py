import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import impute_missing_values, add_stage2_engineered_features

# 1. Load Data
df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
df_clean = impute_missing_values(df_raw)
df_proc = add_stage2_engineered_features(df_clean)

# Baseline feature list
cat_cols = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']
base_num_cols = [
    "Area_sqft", "Number_of_Floors", "Labour_Count", "Planned_Duration",
    "Baseline_Cost", "Planned_Progress_Percent", "Actual_Progress_Percent",
    "Actual_Labour_Count", "Material_Availability_Percent", "Equipment_Availability_Percent",
    "Progress_Gap", "Labour_Deployment_Ratio", "Labour_Shortage", "Labour_Deployment_Gap",
    "Material_Shortage", "Equipment_Shortage", "Progress_Ratio",
    "Resource_Availability_Index", "Resource_Stress_Index", "Project_Size_Duration_Ratio",
    "Area_Per_Floor", "Work_Rate_Deficit"
]

y = df_proc['Expected_Delay_Days']

# Helper to build and evaluate pipeline
def evaluate_pipeline(df_data, num_features, model, random_state=42):
    X_raw = df_data[cat_cols + num_features].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=random_state
    )
    
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_features)
        ],
        remainder='drop'
    )
    
    ct.fit(X_train_raw)
    cat_names = ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feat_names = list(cat_names) + num_features
    
    X_train = pd.DataFrame(ct.transform(X_train_raw), columns=feat_names)
    X_test = pd.DataFrame(ct.transform(X_test_raw), columns=feat_names)
    
    model.fit(X_train, y_train)
    y_pred_train = np.maximum(0, model.predict(X_train))
    y_pred_test = np.maximum(0, model.predict(X_test))
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'gap': train_r2 - test_r2
    }

print("=== BASELINE EVALUATION (ExtraTrees current config) ===")
et_baseline = ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
res_base = evaluate_pipeline(df_proc, base_num_cols, et_baseline)
print(f"Train R2: {res_base['train_r2']:.4f}")
print(f"Test R2:  {res_base['test_r2']:.4f}")
print(f"Test MAE: {res_base['test_mae']:.2f}")
print(f"Test RMSE:{res_base['test_rmse']:.2f}")
print(f"CV R2:    {res_base['cv_mean']:.4f} ± {res_base['cv_std']:.4f}")
