import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
df_base = add_stage2_engineered_features(df_clean)

# Add enhanced domain interaction features
def add_enhanced_features(df):
    df_feat = df.copy()
    
    # Schedule Delay Prorated
    df_feat['Schedule_Delay_Prorated'] = np.where(
        df_feat['Planned_Progress_Percent'] > 0,
        (df_feat['Progress_Gap'] / df_feat['Planned_Progress_Percent'].clip(lower=1.0)) * df_feat['Planned_Duration'],
        0.0
    )
    df_feat['Schedule_Delay_Prorated'] = np.maximum(0, df_feat['Schedule_Delay_Prorated']).fillna(0.0)
    
    # Progress Rate Ratio (Actual daily rate vs Planned daily rate)
    planned_daily = df_feat['Planned_Progress_Percent'] / df_feat['Planned_Duration'].clip(lower=1.0)
    actual_daily = df_feat['Actual_Progress_Percent'] / df_feat['Planned_Duration'].clip(lower=1.0)
    df_feat['Daily_Progress_Ratio'] = (actual_daily / planned_daily.replace(0, np.nan)).fillna(1.0)
    
    # Resource Availability Interaction
    m_ratio = df_feat['Material_Availability_Percent'] / 100.0
    e_ratio = df_feat['Equipment_Availability_Percent'] / 100.0
    l_ratio = np.clip(df_feat['Labour_Deployment_Ratio'], 0, 2)
    df_feat['Combined_Resource_Factor'] = m_ratio * e_ratio * l_ratio
    
    # Labour Progress Interaction
    df_feat['Labour_Progress_Impact'] = l_ratio * df_feat['Actual_Progress_Percent']
    
    # Material & Equipment Progress Impact
    df_feat['Material_Progress_Impact'] = m_ratio * df_feat['Actual_Progress_Percent']
    df_feat['Equipment_Progress_Impact'] = e_ratio * df_feat['Actual_Progress_Percent']
    
    # Total Shortage Index
    df_feat['Total_Resource_Shortage'] = df_feat['Material_Shortage'] + df_feat['Equipment_Shortage'] + (df_feat['Labour_Deployment_Gap'] * 100.0)
    
    # Project Scale & Complexity
    df_feat['Total_Floor_Area'] = df_feat['Area_sqft'] * df_feat['Number_of_Floors']
    df_feat['Cost_Per_Sqft'] = df_feat['Baseline_Cost'] / df_feat['Area_sqft'].replace(0, np.nan)
    df_feat['Cost_Per_Sqft'] = df_feat['Cost_Per_Sqft'].fillna(0.0)
    
    return df_feat

df_enhanced = add_enhanced_features(df_base)

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

enhanced_num_cols = base_num_cols + [
    "Schedule_Delay_Prorated", "Daily_Progress_Ratio", "Combined_Resource_Factor",
    "Labour_Progress_Impact", "Material_Progress_Impact", "Equipment_Progress_Impact",
    "Total_Resource_Shortage", "Total_Floor_Area", "Cost_Per_Sqft"
]

y = df_enhanced['Expected_Delay_Days']

def evaluate_model(df_data, num_cols, model, name):
    X_raw = df_data[cat_cols + num_cols].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    
    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ],
        remainder='drop'
    )
    
    ct.fit(X_train_raw)
    cat_names = ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feat_names = list(cat_names) + num_cols
    
    X_train = pd.DataFrame(ct.transform(X_train_raw), columns=feat_names)
    X_test = pd.DataFrame(ct.transform(X_test_raw), columns=feat_names)
    
    model.fit(X_train, y_train)
    y_pred_tr = np.maximum(0, model.predict(X_train))
    y_pred_te = np.maximum(0, model.predict(X_test))
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    train_r2 = r2_score(y_train, y_pred_tr)
    test_r2 = r2_score(y_test, y_pred_te)
    mae = mean_absolute_error(y_test, y_pred_te)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_te))
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = train_r2 - test_r2
    
    print(f"{name:35s} | Train R2: {train_r2:.4f} | CV R2: {cv_mean:.4f} ± {cv_std:.4f} | Test R2: {test_r2:.4f} ({test_r2*100:.2f}%) | Test MAE: {mae:.2f} | Test RMSE: {rmse:.2f} | Gap: {gap:.4f}")
    return {
        'Name': name,
        'Train_R2': train_r2,
        'CV_R2': cv_mean,
        'CV_Std': cv_std,
        'Test_R2': test_r2,
        'Test_MAE': mae,
        'Test_RMSE': rmse,
        'Gap': gap
    }

print("\n==========================================================================")
print("PART 1: TESTING MODELS ON BASE FEATURES (22 NUMERICAL + 4 CATEGORICAL)")
print("==========================================================================")
models = [
    ("Extra Trees (Current Baseline)", ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)),
    ("Extra Trees (Depth=12, split=6)", ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=6, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("Random Forest (Baseline)", RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=5, min_samples_leaf=10, random_state=42)),
    ("XGBoost (Current Baseline)", xgb.XGBRegressor(n_estimators=350, learning_rate=0.02, max_depth=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=-1, objective='reg:squarederror')),
    ("XGBoost (n_est=400, lr=0.03, d=4)", xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, objective='reg:squarederror')),
    ("XGBoost (n_est=500, lr=0.025, d=5)", xgb.XGBRegressor(n_estimators=500, learning_rate=0.025, max_depth=5, min_child_weight=3, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, random_state=42, n_jobs=-1, objective='reg:squarederror')),
]

for name, model in models:
    evaluate_model(df_base, base_num_cols, model, name)

print("\n==========================================================================")
print("PART 2: TESTING MODELS ON ENHANCED DOMAIN INTERACTION FEATURES")
print("==========================================================================")
for name, model in models:
    evaluate_model(df_enhanced, enhanced_num_cols, model, name)
