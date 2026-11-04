import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import impute_missing_values, add_stage2_engineered_features

# Load & Prepare Data
df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
df_clean = impute_missing_values(df_raw)
df_proc = add_stage2_engineered_features(df_clean)

# Add selected legitimate domain features without target leakage
df_proc['Work_Deficit_Days'] = (df_proc['Progress_Gap'] / 100.0) * df_proc['Planned_Duration']
df_proc['Planned_Work_Rate'] = df_proc['Planned_Progress_Percent'] / df_proc['Planned_Duration'].replace(0, np.nan)
df_proc['Planned_Work_Rate'] = df_proc['Planned_Work_Rate'].fillna(0.0)
df_proc['Actual_Work_Rate'] = df_proc['Actual_Progress_Percent'] / df_proc['Planned_Duration'].replace(0, np.nan)
df_proc['Actual_Work_Rate'] = df_proc['Actual_Work_Rate'].fillna(0.0)
df_proc['Total_Resource_Shortage'] = df_proc['Material_Shortage'] + df_proc['Equipment_Shortage'] + (df_proc['Labour_Deployment_Gap'] * 100.0)

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
    "Work_Deficit_Days", "Planned_Work_Rate", "Actual_Work_Rate", "Total_Resource_Shortage"
]

y = df_proc['Expected_Delay_Days']

# Build train/test splits (80/20) using fixed random_state 42
X_raw_base = df_proc[cat_cols + base_num_cols]
X_raw_enh = df_proc[cat_cols + enhanced_num_cols]

ct_base = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', base_num_cols)
    ],
    remainder='drop'
)
X_tr_b_raw, X_te_b_raw, y_train, y_test = train_test_split(X_raw_base, y, test_size=0.2, random_state=42)
ct_base.fit(X_tr_b_raw)
cat_names_b = ct_base.named_transformers_['cat'].get_feature_names_out(cat_cols)
feat_names_b = list(cat_names_b) + base_num_cols
X_tr_b = pd.DataFrame(ct_base.transform(X_tr_b_raw), columns=feat_names_b)
X_te_b = pd.DataFrame(ct_base.transform(X_te_b_raw), columns=feat_names_b)

ct_enh = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', enhanced_num_cols)
    ],
    remainder='drop'
)
X_tr_e_raw, X_te_e_raw, _, _ = train_test_split(X_raw_enh, y, test_size=0.2, random_state=42)
ct_enh.fit(X_tr_e_raw)
cat_names_e = ct_enh.named_transformers_['cat'].get_feature_names_out(cat_cols)
feat_names_e = list(cat_names_e) + enhanced_num_cols
X_tr_e = pd.DataFrame(ct_enh.transform(X_tr_e_raw), columns=feat_names_e)
X_te_e = pd.DataFrame(ct_enh.transform(X_te_e_raw), columns=feat_names_e)

print(f"Data Prepared. Train shape: {X_tr_b.shape}, Test shape: {X_te_b.shape}")

# Define hyperparameter search spaces
param_grids = {
    "Extra Trees": (
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 12, 14, 16],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 3]
        }
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            'n_estimators': [200, 300, 400],
            'max_depth': [8, 10, 12, 14],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 3]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            'n_estimators': [150, 250, 350],
            'learning_rate': [0.02, 0.03, 0.05],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9]
        }
    ),
    "HistGradientBoosting": (
        HistGradientBoostingRegressor(random_state=42),
        {
            'max_iter': [200, 300, 400],
            'learning_rate': [0.02, 0.03, 0.05],
            'max_depth': [3, 4, 5],
            'min_samples_leaf': [10, 15, 20]
        }
    ),
    "XGBoost": (
        xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
        {
            'n_estimators': [250, 350, 450],
            'learning_rate': [0.015, 0.02, 0.03],
            'max_depth': [3, 4, 5],
            'min_child_weight': [2, 3, 4],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.5, 1.0],
            'reg_lambda': [1.0, 1.5]
        }
    )
}

# Run GridSearch for each model on both base features and enhanced features
results = []

def tune_and_evaluate(model_name, estimator, grid, X_tr, X_te, feat_set_name):
    print(f"\nTuning {model_name} on {feat_set_name}...")
    gs = GridSearchCV(estimator, grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    gs.fit(X_tr, y_train)
    
    best_model = gs.best_estimator_
    best_cv_r2 = gs.best_score_
    
    y_pred_tr = np.maximum(0, best_model.predict(X_tr))
    y_pred_te = np.maximum(0, best_model.predict(X_te))
    
    tr_r2 = r2_score(y_train, y_pred_tr)
    te_r2 = r2_score(y_test, y_pred_te)
    mae = mean_absolute_error(y_test, y_pred_te)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_te))
    gap = tr_r2 - te_r2
    
    res = {
        'Model': model_name,
        'Feature_Set': feat_set_name,
        'Train_R2': tr_r2,
        'CV_R2': best_cv_r2,
        'Test_R2': te_r2,
        'Test_MAE': mae,
        'Test_RMSE': rmse,
        'Gap': gap,
        'Best_Params': gs.best_params_
    }
    print(f"--> {model_name} [{feat_set_name}]: CV R2={best_cv_r2:.4f}, Test R2={te_r2:.4f} ({te_r2*100:.2f}%), MAE={mae:.2f}d, RMSE={rmse:.2f}d | Best Params: {gs.best_params_}")
    return res

for name, (est, grid) in param_grids.items():
    r_b = tune_and_evaluate(name, est, grid, X_tr_b, X_te_b, "Base Features")
    results.append(r_b)
    
    r_e = tune_and_evaluate(name, est, grid, X_tr_e, X_te_e, "Enhanced Features")
    results.append(r_e)

df_all = pd.DataFrame(results)
print("\n==========================================================================")
print("FINAL SUMMARY OF ALL TUNED MODELS")
print("==========================================================================")
print(df_all[['Model', 'Feature_Set', 'Train_R2', 'CV_R2', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'Gap']].to_string(index=False))

df_all.to_csv("outputs/all_models_tuning_results.csv", index=False)
