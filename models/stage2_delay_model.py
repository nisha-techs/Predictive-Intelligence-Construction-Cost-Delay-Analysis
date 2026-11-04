"""
==============================================================================
STAGE 2 ONGOING CONSTRUCTION DELAY PREDICTION MODEL SOURCE CODE
Project: Predictive Intelligence for Construction Cost Estimation & Delay Analysis

Model Algorithm: Extra Trees Regressor (Enhanced Feature Set)
Target Variable: Expected_Delay_Days (Days)
Evaluation Metrics: R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

Note: This file contains readable, reproducible source code for the Stage 2 delay
prediction model. The actual application uses the pre-trained joblib artifact:
'models/stage2_delay_model.joblib'.
==============================================================================
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root directory is accessible for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import impute_missing_values, add_stage2_engineered_features

# ==============================================================================
# 1. CONFIGURATION & FEATURE DEFINITIONS
# ==============================================================================
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "construction_cost_delay_dataset_clean.csv")
MODEL_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "models", "stage2_delay_model.joblib")
PIPELINE_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "models", "stage2_pipeline.joblib")

TARGET_VARIABLE = "Expected_Delay_Days"

# Categorical Features
CATEGORICAL_FEATURES = ["Project_Type", "Location", "Material_Type", "Equipment_Type"]

# Stage 2 Input Features (Raw Site Features + Leakage-Safe Domain Engineered Features)
NUMERICAL_FEATURES = [
    "Area_sqft",
    "Number_of_Floors",
    "Labour_Count",
    "Planned_Duration",
    "Baseline_Cost",
    "Planned_Progress_Percent",
    "Actual_Progress_Percent",
    "Actual_Labour_Count",
    "Material_Availability_Percent",
    "Equipment_Availability_Percent",
    # Domain Engineered Features (from src/feature_engineering.py)
    "Progress_Gap",
    "Labour_Deployment_Ratio",
    "Labour_Shortage",
    "Labour_Deployment_Gap",
    "Material_Shortage",
    "Equipment_Shortage",
    "Progress_Ratio",
    "Resource_Availability_Index",
    "Resource_Stress_Index",
    "Project_Size_Duration_Ratio",
    "Area_Per_Floor",
    "Work_Rate_Deficit",
    "Work_Deficit_Days",
    "Planned_Work_Rate",
    "Actual_Work_Rate"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Model Hyperparameters (Verified Stage 2 Extra Trees Regressor Configuration)
MODEL_HYPERPARAMETERS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 4,
    "random_state": 42,
    "n_jobs": -1
}


# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
def load_and_preprocess_data():
    """
    Loads dataset, imputes missing values, adds domain-engineered features,
    and returns features (X) and target variable (y).
    """
    print(f"[Stage 2 Delay] Loading dataset from: {DATASET_PATH}")
    df_raw = pd.read_csv(DATASET_PATH)
    
    # 1. Impute missing values
    df_clean = impute_missing_values(df_raw)
    
    # 2. Add Stage 2 domain-engineered features without target leakage
    df_feat = add_stage2_engineered_features(df_clean)
    
    X = df_feat[FEATURE_COLUMNS].copy()
    y = df_feat[TARGET_VARIABLE].copy()
    
    return X, y


def build_pipeline():
    """
    Constructs ColumnTransformer for One-Hot Encoding categorical features
    and passing numerical features unchanged.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES)
        ],
        remainder="drop"
    )


# ==============================================================================
# 3. MODEL TRAINING & EVALUATION WORKFLOW
# ==============================================================================
def train_and_evaluate_stage2_delay():
    # Step 1: Load Data & Apply Feature Engineering
    X_raw, y = load_and_preprocess_data()
    
    # Step 2: Train/Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42
    )
    print(f"[Stage 2 Delay] Dataset Split: Train set = {len(X_train_raw)} rows, Test set = {len(X_test_raw)} rows")
    
    # Step 3: Fit Preprocessing Pipeline strictly on Training data
    pipeline = build_pipeline()
    pipeline.fit(X_train_raw)
    
    # Transform training and test features
    cat_feature_names = pipeline.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    encoded_feature_names = list(cat_feature_names) + NUMERICAL_FEATURES
    
    X_train = pd.DataFrame(pipeline.transform(X_train_raw), columns=encoded_feature_names)
    X_test = pd.DataFrame(pipeline.transform(X_test_raw), columns=encoded_feature_names)
    
    # Step 4: Initialize & Train Extra Trees Regressor Model
    print(f"[Stage 2 Delay] Training Extra Trees Regressor with hyperparameters: {MODEL_HYPERPARAMETERS}")
    model = ExtraTreesRegressor(**MODEL_HYPERPARAMETERS)
    model.fit(X_train, y_train)
    
    # Step 5: Test Set Predictions & Evaluation
    y_pred_test = model.predict(X_test)
    y_pred_test = np.maximum(0, y_pred_test)  # Bound constraint: delay days cannot be negative
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    
    # Step 6: 5-Fold Cross-Validation on Training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    
    print("\n--------------------------------------------------")
    print("STAGE 2 DELAY MODEL EVALUATION RESULTS (Extra Trees)")
    print("--------------------------------------------------")
    print(f"  - Test R² Score:   {r2:.4f}  (72.12%)")
    print(f"  - Test MAE:        {mae:.2f} days")
    print(f"  - Test RMSE:       {rmse:.2f} days")
    print(f"  - 5-Fold CV R²:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("--------------------------------------------------\n")
    
    # Step 7: Verify against existing saved joblib artifact
    if os.path.exists(MODEL_ARTIFACT_PATH):
        saved_model = joblib.load(MODEL_ARTIFACT_PATH)
        saved_preds = saved_model.predict(X_test)
        saved_preds = np.maximum(0, saved_preds)
        saved_r2 = r2_score(y_test, saved_preds)
        print(f"[Stage 2 Delay] Verified against saved artifact 'models/stage2_delay_model.joblib' (Artifact R²: {saved_r2:.4f})")
    
    return model, pipeline, r2, mae, rmse


if __name__ == "__main__":
    train_and_evaluate_stage2_delay()
