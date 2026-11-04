import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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

# Preprocessing Pipelines
X_raw_base = df_proc[cat_cols + base_num_cols]
X_raw_enh = df_proc[cat_cols + enhanced_num_cols]

X_tr_b_raw, X_te_b_raw, y_train, y_test = train_test_split(X_raw_base, y, test_size=0.2, random_state=42)
ct_base = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', base_num_cols)
    ],
    remainder='drop'
)
ct_base.fit(X_tr_b_raw)
cat_names_b = ct_base.named_transformers_['cat'].get_feature_names_out(cat_cols)
feat_names_b = list(cat_names_b) + base_num_cols
X_tr_b = pd.DataFrame(ct_base.transform(X_tr_b_raw), columns=feat_names_b)
X_te_b = pd.DataFrame(ct_base.transform(X_te_b_raw), columns=feat_names_b)

X_tr_e_raw, X_te_e_raw, _, _ = train_test_split(X_raw_enh, y, test_size=0.2, random_state=42)
ct_enh = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', enhanced_num_cols)
    ],
    remainder='drop'
)
ct_enh.fit(X_tr_e_raw)
cat_names_e = ct_enh.named_transformers_['cat'].get_feature_names_out(cat_cols)
feat_names_e = list(cat_names_e) + enhanced_num_cols
X_tr_e = pd.DataFrame(ct_enh.transform(X_tr_e_raw), columns=feat_names_e)
X_te_e = pd.DataFrame(ct_enh.transform(X_te_e_raw), columns=feat_names_e)

# 1. Baseline Old Model
m_old = ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
m_old.fit(X_tr_b, y_train)
y_pred_old = np.maximum(0, m_old.predict(X_te_b))
cv_old = cross_val_score(m_old, X_tr_b, y_train, cv=5, scoring='r2', n_jobs=-1)

r2_old = r2_score(y_test, y_pred_old)
mae_old = mean_absolute_error(y_test, y_pred_old)
rmse_old = np.sqrt(mean_squared_error(y_test, y_pred_old))
cv_r2_old = cv_old.mean()

print("\nOLD MODEL (Extra Trees Baseline):")
print(f"  R² = {r2_old:.4f} ({r2_old*100:.2f}%)")
print(f"  MAE = {mae_old:.2f} days")
print(f"  RMSE = {rmse_old:.2f} days")
print(f"  CV R² = {cv_r2_old:.4f} ({cv_r2_old*100:.2f}%)")

# 2. Enhanced Model Option A: Extra Trees (Baseline params on Enhanced Features)
m_new_a = ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
m_new_a.fit(X_tr_e, y_train)
y_pred_new_a = np.maximum(0, m_new_a.predict(X_te_e))
cv_new_a = cross_val_score(m_new_a, X_tr_e, y_train, cv=5, scoring='r2', n_jobs=-1)

r2_new_a = r2_score(y_test, y_pred_new_a)
mae_new_a = mean_absolute_error(y_test, y_pred_new_a)
rmse_new_a = np.sqrt(mean_squared_error(y_test, y_pred_new_a))
cv_r2_new_a = cv_new_a.mean()

print("\nNEW MODEL OPTION A (Extra Trees + Enhanced Domain Features):")
print(f"  R² = {r2_new_a:.4f} ({r2_new_a*100:.2f}%)")
print(f"  MAE = {mae_new_a:.2f} days")
print(f"  RMSE = {rmse_new_a:.2f} days")
print(f"  CV R² = {cv_r2_new_a:.4f} ({cv_r2_new_a*100:.2f}%)")

print("\nIMPROVEMENT OPTION A:")
print(f"  R² improvement = +{(r2_new_a - r2_old)*100:.2f}% (from {r2_old*100:.2f}% to {r2_new_a*100:.2f}%)")
print(f"  MAE improvement = -{mae_old - mae_new_a:.2f} days (from {mae_old:.2f}d to {mae_new_a:.2f}d)")
print(f"  RMSE improvement = -{rmse_old - rmse_new_a:.2f} days (from {rmse_old:.2f}d to {rmse_new_a:.2f}d)")
print(f"  CV R² improvement = +{(cv_r2_new_a - cv_r2_old)*100:.2f}% (from {cv_r2_old*100:.2f}% to {cv_r2_new_a*100:.2f}%)")

# 3. Enhanced Model Option B: Extra Trees (Regularized & Tuned: max_depth=14, min_samples_split=4, min_samples_leaf=2)
m_new_b = ExtraTreesRegressor(n_estimators=350, max_depth=14, min_samples_split=4, min_samples_leaf=2, random_state=42, n_jobs=-1)
m_new_b.fit(X_tr_e, y_train)
y_pred_new_b = np.maximum(0, m_new_b.predict(X_te_e))
cv_new_b = cross_val_score(m_new_b, X_tr_e, y_train, cv=5, scoring='r2', n_jobs=-1)

r2_new_b = r2_score(y_test, y_pred_new_b)
mae_new_b = mean_absolute_error(y_test, y_pred_new_b)
rmse_new_b = np.sqrt(mean_squared_error(y_test, y_pred_new_b))
cv_r2_new_b = cv_new_b.mean()

print("\nNEW MODEL OPTION B (Regularized Extra Trees + Enhanced Domain Features):")
print(f"  R² = {r2_new_b:.4f} ({r2_new_b*100:.2f}%)")
print(f"  MAE = {mae_new_b:.2f} days")
print(f"  RMSE = {rmse_new_b:.2f} days")
print(f"  CV R² = {cv_r2_new_b:.4f} ({cv_r2_new_b*100:.2f}%)")

print("\nIMPROVEMENT OPTION B:")
print(f"  R² improvement = +{(r2_new_b - r2_old)*100:.2f}% (from {r2_old*100:.2f}% to {r2_new_b*100:.2f}%)")
print(f"  MAE improvement = -{mae_old - mae_new_b:.2f} days (from {mae_old:.2f}d to {mae_new_b:.2f}d)")
print(f"  RMSE improvement = -{rmse_old - rmse_new_b:.2f} days (from {rmse_old:.2f}d to {rmse_new_b:.2f}d)")
print(f"  CV R² improvement = +{(cv_r2_new_b - cv_r2_old)*100:.2f}% (from {cv_r2_old*100:.2f}% to {cv_r2_new_b*100:.2f}%)")
