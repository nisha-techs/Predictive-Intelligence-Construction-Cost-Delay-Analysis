import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import impute_missing_values, add_stage2_engineered_features

df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
df_clean = impute_missing_values(df_raw)
df_proc = add_stage2_engineered_features(df_clean)

# Add selected legitimate domain features
df_proc['Work_Deficit_Days'] = (df_proc['Progress_Gap'] / 100.0) * df_proc['Planned_Duration']
df_proc['Planned_Work_Rate'] = df_proc['Planned_Progress_Percent'] / df_proc['Planned_Duration'].replace(0, np.nan)
df_proc['Planned_Work_Rate'] = df_proc['Planned_Work_Rate'].fillna(0.0)
df_proc['Actual_Work_Rate'] = df_proc['Actual_Progress_Percent'] / df_proc['Planned_Duration'].replace(0, np.nan)
df_proc['Actual_Work_Rate'] = df_proc['Actual_Work_Rate'].fillna(0.0)

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
    "Work_Deficit_Days", "Planned_Work_Rate", "Actual_Work_Rate"
]

y = df_proc['Expected_Delay_Days']

def run_eval(model, num_cols, name):
    X_raw = df_proc[cat_cols + num_cols]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
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
    
    tr_r2 = r2_score(y_train, y_pred_tr)
    te_r2 = r2_score(y_test, y_pred_te)
    mae = mean_absolute_error(y_test, y_pred_te)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_te))
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = tr_r2 - te_r2
    
    return {
        'Model': name,
        'Train_R2': tr_r2,
        'CV_R2': cv_mean,
        'CV_Std': cv_std,
        'Test_R2': te_r2,
        'Test_MAE': mae,
        'Test_RMSE': rmse,
        'Gap': gap
    }

models = [
    ("Extra Trees (Current)", ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)),
    ("Extra Trees (Tuned d=12, s=6)", ExtraTreesRegressor(n_estimators=350, max_depth=12, min_samples_split=6, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("Extra Trees (Tuned d=14, s=4)", ExtraTreesRegressor(n_estimators=350, max_depth=14, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("Random Forest (Baseline)", RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)),
    ("Random Forest (Tuned)", RandomForestRegressor(n_estimators=350, max_depth=11, min_samples_split=6, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=4, subsample=0.8, random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=5, min_samples_leaf=15, random_state=42)),
    ("XGBoost (Baseline)", xgb.XGBRegressor(n_estimators=350, learning_rate=0.02, max_depth=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42, n_jobs=-1, objective='reg:squarederror')),
    ("XGBoost (Tuned)", xgb.XGBRegressor(n_estimators=400, learning_rate=0.025, max_depth=4, min_child_weight=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.8, reg_lambda=1.5, random_state=42, n_jobs=-1, objective='reg:squarederror'))
]

print("=== BASE FEATURES EVALUATION ===")
res_base = []
for name, m in models:
    r = run_eval(m, base_num_cols, name)
    res_base.append(r)
df_b = pd.DataFrame(res_base)
print(df_b[['Model', 'Train_R2', 'CV_R2', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'Gap']].to_string(index=False))

print("\n=== ENHANCED DOMAIN FEATURES EVALUATION (+Work_Deficit_Days, Planned/Actual Work Rate) ===")
res_enh = []
for name, m in models:
    r = run_eval(m, enhanced_num_cols, name)
    res_enh.append(r)
df_e = pd.DataFrame(res_enh)
print(df_e[['Model', 'Train_R2', 'CV_R2', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'Gap']].to_string(index=False))
