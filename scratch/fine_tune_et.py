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

# Legitimate domain features
df_proc['Work_Deficit_Days'] = (df_proc['Progress_Gap'] / 100.0) * df_proc['Planned_Duration']
df_proc['Planned_Work_Rate'] = df_proc['Planned_Progress_Percent'] / df_proc['Planned_Duration'].replace(0, np.nan)
df_proc['Planned_Work_Rate'] = df_proc['Planned_Work_Rate'].fillna(0.0)
df_proc['Actual_Work_Rate'] = df_proc['Actual_Progress_Percent'] / df_proc['Planned_Duration'].replace(0, np.nan)
df_proc['Actual_Work_Rate'] = df_proc['Actual_Work_Rate'].fillna(0.0)

cat_cols = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']

num_cols_opt1 = [
    "Area_sqft", "Number_of_Floors", "Labour_Count", "Planned_Duration",
    "Baseline_Cost", "Planned_Progress_Percent", "Actual_Progress_Percent",
    "Actual_Labour_Count", "Material_Availability_Percent", "Equipment_Availability_Percent",
    "Progress_Gap", "Labour_Deployment_Ratio", "Labour_Shortage", "Labour_Deployment_Gap",
    "Material_Shortage", "Equipment_Shortage", "Progress_Ratio",
    "Resource_Availability_Index", "Resource_Stress_Index", "Project_Size_Duration_Ratio",
    "Area_Per_Floor", "Work_Rate_Deficit",
    "Work_Deficit_Days", "Planned_Work_Rate", "Actual_Work_Rate"
]

y = df_proc['Expected_Delay_Days']

X_raw = df_proc[cat_cols + num_cols_opt1]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

ct = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols_opt1)
    ],
    remainder='drop'
)

ct.fit(X_train_raw)
cat_names = ct.named_transformers_['cat'].get_feature_names_out(cat_cols)
feat_names = list(cat_names) + num_cols_opt1

X_train = pd.DataFrame(ct.transform(X_train_raw), columns=feat_names)
X_test = pd.DataFrame(ct.transform(X_test_raw), columns=feat_names)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

results = []

for n_est in [300, 350, 400]:
    for depth in [12, 13, 14, 15]:
        for split in [3, 4, 5, 6]:
            for leaf in [1, 2]:
                et = ExtraTreesRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_split=split,
                    min_samples_leaf=leaf,
                    random_state=42,
                    n_jobs=-1
                )
                et.fit(X_train, y_train)
                
                y_pred_tr = np.maximum(0, et.predict(X_train))
                y_pred_te = np.maximum(0, et.predict(X_test))
                
                cv = cross_val_score(et, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
                
                tr_r2 = r2_score(y_train, y_pred_tr)
                te_r2 = r2_score(y_test, y_pred_te)
                mae = mean_absolute_error(y_test, y_pred_te)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_te))
                cv_m = cv.mean()
                cv_s = cv.std()
                gap = tr_r2 - te_r2
                
                results.append({
                    'n_est': n_est,
                    'depth': depth,
                    'split': split,
                    'leaf': leaf,
                    'Train_R2': tr_r2,
                    'CV_R2': cv_m,
                    'CV_Std': cv_s,
                    'Test_R2': te_r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Gap': gap
                })

df_res = pd.DataFrame(results)
df_res = df_res.sort_values(by='CV_R2', ascending=False)
print("\nTop 10 Configurations by 5-Fold CV R2:")
print(df_res.head(10).to_string(index=False))

print("\nTop 10 Configurations by Test R2:")
print(df_res.sort_values(by='Test_R2', ascending=False).head(10).to_string(index=False))
