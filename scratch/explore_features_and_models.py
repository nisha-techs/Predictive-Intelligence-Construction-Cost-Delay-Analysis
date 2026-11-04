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

df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
df_clean = impute_missing_values(df_raw)
df_base = add_stage2_engineered_features(df_clean)

# Add enhanced domain interaction features
def add_enhanced_features(df):
    df_feat = df.copy()
    
    # Prorated schedule delay based on current progress rate deficit
    # If planned progress = 50%, actual = 30%, gap = 20%. Estimated delay = (20 / 50) * Planned_Duration
    planned_prog = df_feat['Planned_Progress_Percent'].replace(0, np.nan)
    actual_prog = df_feat['Actual_Progress_Percent'].replace(0, np.nan)
    planned_dur = df_feat['Planned_Duration'].replace(0, np.nan)
    
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
    
    # Resource Availability Interaction (Material * Equipment * Labour)
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

def run_experiment(df_data, num_cols, model, name="Model", random_state=42):
    X_raw = df_data[cat_cols + num_cols].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=random_state
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
    
    return {
        'Name': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Test_MAE': mae,
        'Test_RMSE': rmse,
        'CV_R2': cv_mean,
        'CV_Std': cv_std,
        'Gap': train_r2 - test_r2
    }

models_to_test = {
    "ExtraTrees (Current)": ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1),
    "ExtraTrees (Tuned Regularized)": ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_split=6, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "RandomForest (Default/Tuned)": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_samples_split=5, min_samples_leaf=3, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=5, min_samples_leaf=10, random_state=42),
    "XGBoost (Current)": xgb.XGBRegressor(n_estimators=350, learning_rate=0.02, max_depth=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42, objective='reg:squarederror'),
    "XGBoost (Tuned 1)": xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, objective='reg:squarederror'),
    "XGBoost (Tuned 2)": xgb.XGBRegressor(n_estimators=500, learning_rate=0.025, max_depth=5, min_child_weight=3, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, random_state=42, objective='reg:squarederror')
}

print("\n--- EVALUATION WITH BASE FEATURES ---")
res_base = []
for name, model in models_to_test.items():
    r = run_experiment(df_base, base_num_cols, model, name=name)
    res_base.append(r)
df_res_base = pd.DataFrame(res_base)
print(df_res_base[['Name', 'Train_R2', 'CV_R2', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'Gap']].to_string(index=False))

print("\n--- EVALUATION WITH ENHANCED DOMAIN INTERACTION FEATURES ---")
res_enh = []
for name, model in models_to_test.items():
    r = run_experiment(df_enhanced, enhanced_num_cols, model, name=name)
    res_enh.append(r)
df_res_enh = pd.DataFrame(res_enh)
print(df_res_enh[['Name', 'Train_R2', 'CV_R2', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'Gap']].to_string(index=False))
