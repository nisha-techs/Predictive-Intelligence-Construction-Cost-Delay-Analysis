import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import local feature engineering helper
try:
    from src.feature_engineering import impute_missing_values, add_stage2_engineered_features
except ModuleNotFoundError:
    from feature_engineering import impute_missing_values, add_stage2_engineered_features

def get_feature_definitions():
    """
    Returns strict feature lists for Stage 1, Stage 2 Delay, and Stage 2 Risk.
    """
    stage1_raw_features = [
        'Project_Type', 'Location', 'Area_sqft', 'Number_of_Floors',
        'Material_Type', 'Material_Cost', 'Labour_Cost', 'Labour_Count',
        'Equipment_Type', 'Equipment_Cost', 'Planned_Duration'
    ]
    
    stage2_raw_features = [
        'Project_Type', 'Location', 'Area_sqft', 'Number_of_Floors',
        'Material_Type', 'Labour_Count', 'Equipment_Type', 'Planned_Duration',
        'Baseline_Cost', 'Planned_Progress_Percent', 'Actual_Progress_Percent',
        'Actual_Labour_Count', 'Material_Availability_Percent', 'Equipment_Availability_Percent',
        'Progress_Gap', 'Labour_Deployment_Ratio', 'Labour_Shortage', 'Labour_Deployment_Gap',
        'Material_Shortage', 'Equipment_Shortage', 'Progress_Ratio',
        'Resource_Availability_Index', 'Resource_Stress_Index',
        'Project_Size_Duration_Ratio', 'Area_Per_Floor', 'Work_Rate_Deficit',
        'Work_Deficit_Days', 'Planned_Work_Rate', 'Actual_Work_Rate'
    ]
    
    return stage1_raw_features, stage2_raw_features

def build_column_transformer(categorical_cols, numeric_cols):
    """
    Builds a scikit-learn ColumnTransformer for One-Hot Encoding categoricals and passing numerics through.
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ],
        remainder='drop'
    )

def process_and_validate_pipeline(csv_path="dataset/construction_cost_delay_dataset_clean.csv"):
    """
    Loads dataset, applies leakage-safe preprocessing and train/test splits.
    Returns processed data dictionaries and summary dataframes.
    """
    df_raw = pd.read_csv(csv_path)
    
    # 1. Impute missing values
    df_clean = impute_missing_values(df_raw)
    
    # 2. Add engineered features
    df_proc = add_stage2_engineered_features(df_clean)
    
    # Preserve Project_ID separately
    project_ids = df_proc['Project_ID']
    
    # Extract Targets
    y_stage1 = df_proc['Total_Construction_Cost']
    y_stage2_delay = df_proc['Expected_Delay_Days']
    y_stage2_risk = df_proc['Delay_Risk']
    
    stage1_cols, stage2_cols = get_feature_definitions()
    
    X_stage1_raw = df_proc[stage1_cols]
    X_stage2_raw = df_proc[stage2_cols]
    
    # Categorical and Numeric lists
    cat_cols_s1 = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']
    num_cols_s1 = [c for c in stage1_cols if c not in cat_cols_s1]
    
    cat_cols_s2 = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']
    num_cols_s2 = [c for c in stage2_cols if c not in cat_cols_s2]
    
    # Train / Test split before encoding (80/20 split)
    X_s1_train_raw, X_s1_test_raw, y_s1_train, y_s1_test = train_test_split(
        X_stage1_raw, y_stage1, test_size=0.2, random_state=42
    )
    
    X_s2_train_raw, X_s2_test_raw, y_s2_delay_train, y_s2_delay_test, y_s2_risk_train, y_s2_risk_test = train_test_split(
        X_stage2_raw, y_stage2_delay, y_stage2_risk, test_size=0.2, random_state=42
    )
    
    # Fit Transformers ONLY on Training sets
    ct_s1 = build_column_transformer(cat_cols_s1, num_cols_s1)
    ct_s1.fit(X_s1_train_raw)
    
    ct_s2 = build_column_transformer(cat_cols_s2, num_cols_s2)
    ct_s2.fit(X_s2_train_raw)
    
    # Feature names output
    cat_s1_names = ct_s1.named_transformers_['cat'].get_feature_names_out(cat_cols_s1)
    s1_encoded_feature_names = list(cat_s1_names) + num_cols_s1
    
    cat_s2_names = ct_s2.named_transformers_['cat'].get_feature_names_out(cat_cols_s2)
    s2_encoded_feature_names = list(cat_s2_names) + num_cols_s2
    
    # Transform Train and Test sets
    X_s1_train = pd.DataFrame(ct_s1.transform(X_s1_train_raw), columns=s1_encoded_feature_names)
    X_s1_test = pd.DataFrame(ct_s1.transform(X_s1_test_raw), columns=s1_encoded_feature_names)
    
    X_s2_train = pd.DataFrame(ct_s2.transform(X_s2_train_raw), columns=s2_encoded_feature_names)
    X_s2_test = pd.DataFrame(ct_s2.transform(X_s2_test_raw), columns=s2_encoded_feature_names)
    
    # Validation checks
    assert 'Baseline_Cost' not in X_s1_train.columns, "Baseline_Cost leaked into Stage 1!"
    assert 'Expected_Delay_Days' not in X_s2_train.columns, "Expected_Delay_Days leaked into Stage 2 features!"
    assert 'Delay_Risk' not in X_s2_train.columns, "Delay_Risk leaked into Stage 2 features!"
    assert 'Project_ID' not in X_s1_train.columns and 'Project_ID' not in X_s2_train.columns, "Project_ID leaked into ML features!"
    assert X_s1_train.isnull().sum().sum() == 0, "Null values remain in Stage 1 Train!"
    assert X_s2_train.isnull().sum().sum() == 0, "Null values remain in Stage 2 Train!"
    
    # Create Summaries for export
    os.makedirs('outputs', exist_ok=True)
    
    prep_summary = pd.DataFrame([
        {'Stage': 'Stage 1 (Cost Estimation)', 'Raw Features': len(stage1_cols), 'Encoded Features': len(s1_encoded_feature_names), 'Train Rows': len(X_s1_train), 'Test Rows': len(X_s1_test), 'Target': 'Total_Construction_Cost'},
        {'Stage': 'Stage 2 (Delay Days)', 'Raw Features': len(stage2_cols), 'Encoded Features': len(s2_encoded_feature_names), 'Train Rows': len(X_s2_train), 'Test Rows': len(X_s2_test), 'Target': 'Expected_Delay_Days'},
        {'Stage': 'Stage 2 (Delay Risk)', 'Raw Features': len(stage2_cols), 'Encoded Features': len(s2_encoded_feature_names), 'Train Rows': len(X_s2_train), 'Test Rows': len(X_s2_test), 'Target': 'Delay_Risk'}
    ])
    prep_summary.to_csv('outputs/preprocessing_summary.csv', index=False)
    
    s1_feat_df = pd.DataFrame({'Feature_Name': s1_encoded_feature_names, 'Pipeline_Stage': 'Stage 1'})
    s2_feat_df = pd.DataFrame({'Feature_Name': s2_encoded_feature_names, 'Pipeline_Stage': 'Stage 2'})
    feat_summary = pd.concat([s1_feat_df, s2_feat_df], ignore_index=True)
    feat_summary.to_csv('outputs/feature_summary.csv', index=False)
    
    return {
        's1_feature_names': s1_encoded_feature_names,
        's2_feature_names': s2_encoded_feature_names,
        'prep_summary': prep_summary,
        'X_s1_train_shape': X_s1_train.shape,
        'X_s1_test_shape': X_s1_test.shape,
        'X_s2_train_shape': X_s2_train.shape,
        'X_s2_test_shape': X_s2_test.shape,
    }

if __name__ == '__main__':
    results = process_and_validate_pipeline()
    print("Preprocessing and Feature Pipeline Validation Successful!")
    print(f"Stage 1 Feature Count (Encoded): {len(results['s1_feature_names'])}")
    print(f"Stage 2 Feature Count (Encoded): {len(results['s2_feature_names'])}")
    print(f"Stage 1 Train Shape: {results['X_s1_train_shape']}, Test Shape: {results['X_s1_test_shape']}")
    print(f"Stage 2 Train Shape: {results['X_s2_train_shape']}, Test Shape: {results['X_s2_test_shape']}")
