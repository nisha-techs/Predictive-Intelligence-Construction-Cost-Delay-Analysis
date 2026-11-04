"""
==============================================================================
STAGE 2 DELAY RISK MULTI-CLASS CLASSIFICATION MODEL SOURCE CODE
Project: Predictive Intelligence for Construction Cost Estimation & Delay Analysis

Model Algorithm: Random Forest Classifier (with Class Weighting)
Target Variable: Delay_Risk ('Low', 'Medium', 'High')
Evaluation Metrics: Accuracy, Weighted F1 Score, Macro F1 Score, Weighted ROC-AUC

Note: This file contains readable, reproducible source code for the Stage 2 delay risk
classification model. The actual application uses the pre-trained joblib artifacts:
'models/stage2_risk_model.joblib' and 'models/stage2_risk_label_encoder.joblib'.
==============================================================================
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Ensure project root directory is accessible for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import impute_missing_values, add_stage2_engineered_features

# ==============================================================================
# 1. CONFIGURATION & FEATURE DEFINITIONS
# ==============================================================================
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "construction_cost_delay_dataset_clean.csv")
MODEL_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "models", "stage2_risk_model.joblib")
ENCODER_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "models", "stage2_risk_label_encoder.joblib")
PIPELINE_ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "models", "stage2_pipeline.joblib")

TARGET_VARIABLE = "Delay_Risk"
RISK_CLASSES = ["Low", "Medium", "High"]

# Categorical Features
CATEGORICAL_FEATURES = ["Project_Type", "Location", "Material_Type", "Equipment_Type"]

# Stage 2 Input Features
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
    "Work_Rate_Deficit"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Model Hyperparameters (Verified Stage 2 Random Forest Classifier Configuration)
MODEL_HYPERPARAMETERS = {
    "n_estimators": 100,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": 42
}


# ==============================================================================
# 2. DATA LOADING & PREPROCESSING
# ==============================================================================
def load_and_preprocess_data():
    """
    Loads dataset, imputes missing values, adds domain-engineered features,
    and returns features (X) and target variable string labels (y).
    """
    print(f"[Stage 2 Risk] Loading dataset from: {DATASET_PATH}")
    df_raw = pd.read_csv(DATASET_PATH)
    
    # 1. Impute missing values
    df_clean = impute_missing_values(df_raw)
    
    # 2. Add Stage 2 domain-engineered features
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
def train_and_evaluate_stage2_risk():
    # Step 1: Load Data & Apply Feature Engineering
    X_raw, y_str = load_and_preprocess_data()
    
    # Step 2: Encode Target Variable using LabelEncoder ['Low', 'Medium', 'High']
    label_encoder = LabelEncoder()
    label_encoder.fit(RISK_CLASSES)
    y_encoded = label_encoder.transform(y_str)
    
    # Step 3: Train/Test Split (80% Train, 20% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.20, random_state=42
    )
    print(f"[Stage 2 Risk] Dataset Split: Train set = {len(X_train_raw)} rows, Test set = {len(X_test_raw)} rows")
    
    # Step 4: Fit Preprocessing Pipeline strictly on Training data
    pipeline = build_pipeline()
    pipeline.fit(X_train_raw)
    
    # Transform training and test features
    cat_feature_names = pipeline.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    encoded_feature_names = list(cat_feature_names) + NUMERICAL_FEATURES
    
    X_train = pd.DataFrame(pipeline.transform(X_train_raw), columns=encoded_feature_names)
    X_test = pd.DataFrame(pipeline.transform(X_test_raw), columns=encoded_feature_names)
    
    # Step 5: Initialize & Train Random Forest Classifier Model
    print(f"[Stage 2 Risk] Training Random Forest Classifier with hyperparameters: {MODEL_HYPERPARAMETERS}")
    model = RandomForestClassifier(**MODEL_HYPERPARAMETERS)
    model.fit(X_train, y_train)
    
    # Step 6: Test Set Predictions & Evaluation
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred_test)
    prec_w = precision_score(y_test, y_pred_test, average="weighted")
    rec_w = recall_score(y_test, y_pred_test, average="weighted")
    f1_w = f1_score(y_test, y_pred_test, average="weighted")
    f1_m = f1_score(y_test, y_pred_test, average="macro")
    roc_auc = roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="weighted")
    
    # Step 7: 5-Fold Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1)
    
    print("\n--------------------------------------------------")
    print("STAGE 2 RISK MODEL EVALUATION RESULTS (Random Forest Classifier)")
    print("--------------------------------------------------")
    print(f"  - Accuracy:             {acc * 100:.2f}%")
    print(f"  - Weighted Precision:   {prec_w * 100:.2f}%")
    print(f"  - Weighted Recall:      {rec_w * 100:.2f}%")
    print(f"  - Weighted F1 Score:    {f1_w:.4f}")
    print(f"  - Macro F1 Score:       {f1_m:.4f}")
    print(f"  - Weighted ROC-AUC:     {roc_auc:.4f}")
    print(f"  - 5-Fold CV F1 Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("--------------------------------------------------\n")
    
    # Detailed Classification Report
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_test)
    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, target_names=RISK_CLASSES))
    
    # Step 8: Verify against existing saved joblib artifact
    if os.path.exists(MODEL_ARTIFACT_PATH) and os.path.exists(ENCODER_ARTIFACT_PATH):
        saved_model = joblib.load(MODEL_ARTIFACT_PATH)
        saved_encoder = joblib.load(ENCODER_ARTIFACT_PATH)
        saved_preds = saved_model.predict(X_test)
        saved_acc = accuracy_score(y_test, saved_preds)
        print(f"[Stage 2 Risk] Verified against saved artifact 'models/stage2_risk_model.joblib' (Artifact Accuracy: {saved_acc * 100:.2f}%)")
    
    return model, label_encoder, pipeline, acc, f1_w, f1_m, roc_auc


if __name__ == "__main__":
    train_and_evaluate_stage2_risk()

