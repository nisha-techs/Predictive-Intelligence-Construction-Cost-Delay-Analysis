import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import impute_missing_values, add_stage2_engineered_features

df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
df_clean = impute_missing_values(df_raw)
df_base = add_stage2_engineered_features(df_clean)

# Check target distribution
print("Target variable 'Expected_Delay_Days' statistics:")
print(df_base['Expected_Delay_Days'].describe())

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

# Check correlations with Expected_Delay_Days
num_df = df_base[base_num_cols + ['Expected_Delay_Days']]
corrs = num_df.corr()['Expected_Delay_Days'].sort_values(key=abs, ascending=False)
print("\nTop Numerical Feature Correlations with Expected_Delay_Days:")
print(corrs)

# Let's test individual candidate feature additions to see which ones improve CV R2 and Test R2
candidates = {}

# Candidate 1: Prorated_Delay_Days = (Progress_Gap / Planned_Progress_Percent) * Planned_Duration
df_test = df_base.copy()
df_test['Prorated_Delay_Days'] = np.where(
    df_test['Planned_Progress_Percent'] > 0,
    (df_test['Progress_Gap'] / df_test['Planned_Progress_Percent'].clip(lower=1.0)) * df_test['Planned_Duration'],
    0.0
).clip(min=0.0)
candidates['Prorated_Delay_Days'] = df_test['Prorated_Delay_Days']

# Candidate 2: Schedule_Performance_Index (SPI) = Actual_Progress_Percent / Planned_Progress_Percent
# Note: Progress_Ratio is already in base features (Actual / Planned).

# Candidate 3: Remaining_Work_Days = Planned_Duration * (100 - Actual_Progress_Percent) / 100
df_test['Remaining_Work_Days'] = df_test['Planned_Duration'] * (100.0 - df_test['Actual_Progress_Percent']) / 100.0
candidates['Remaining_Work_Days'] = df_test['Remaining_Work_Days']

# Candidate 4: Work_Deficit_Days = (Progress_Gap / 100.0) * Planned_Duration
df_test['Work_Deficit_Days'] = (df_test['Progress_Gap'] / 100.0) * df_test['Planned_Duration']
candidates['Work_Deficit_Days'] = df_test['Work_Deficit_Days']

# Candidate 5: Resource_Deficit_Index = Material_Shortage + Equipment_Shortage + Labour_Deployment_Gap * 100
df_test['Resource_Deficit_Index'] = df_test['Material_Shortage'] + df_test['Equipment_Shortage'] + (df_test['Labour_Deployment_Gap'] * 100.0)
candidates['Resource_Deficit_Index'] = df_test['Resource_Deficit_Index']

# Candidate 6: Material_Labour_Stress = Material_Shortage * Labour_Shortage
df_test['Material_Labour_Stress'] = df_test['Material_Shortage'] * np.maximum(0, df_test['Labour_Shortage'])
candidates['Material_Labour_Stress'] = df_test['Material_Labour_Stress']

# Candidate 7: Actual_Work_Rate = Actual_Progress_Percent / Planned_Duration
df_test['Actual_Work_Rate'] = df_test['Actual_Progress_Percent'] / df_test['Planned_Duration'].replace(0, np.nan)
candidates['Actual_Work_Rate'] = df_test['Actual_Work_Rate'].fillna(0.0)

# Candidate 8: Planned_Work_Rate = Planned_Progress_Percent / Planned_Duration
df_test['Planned_Work_Rate'] = df_test['Planned_Progress_Percent'] / df_test['Planned_Duration'].replace(0, np.nan)
candidates['Planned_Work_Rate'] = df_test['Planned_Work_Rate'].fillna(0.0)

# Candidate 9: Area_Labor_Density = Actual_Labour_Count / Area_sqft
df_test['Area_Labor_Density'] = df_test['Actual_Labour_Count'] / df_test['Area_sqft'].replace(0, np.nan)
candidates['Area_Labor_Density'] = df_test['Area_Labor_Density'].fillna(0.0)

# Test baseline model
y = df_base['Expected_Delay_Days']

def eval_features(df_data, num_cols, name):
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
    
    model = ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_split=6, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_te = np.maximum(0, model.predict(X_test))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    cv_mean = cv_scores.mean()
    test_r2 = r2_score(y_test, y_pred_te)
    mae = mean_absolute_error(y_test, y_pred_te)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_te))
    
    print(f"{name:30s} | CV R2: {cv_mean:.4f} | Test R2: {test_r2:.4f} ({test_r2*100:.2f}%) | MAE: {mae:.2f}d | RMSE: {rmse:.2f}d")
    return cv_mean, test_r2, mae, rmse

print("\n--- INDIVIDUAL FEATURE IMPACT (ExtraTrees max_depth=12, min_samples_split=6) ---")
eval_features(df_base, base_num_cols, "Base Features Only")

for feat_name, feat_series in candidates.items():
    df_temp = df_base.copy()
    df_temp[feat_name] = feat_series
    eval_features(df_temp, base_num_cols + [feat_name], f"+ {feat_name}")
