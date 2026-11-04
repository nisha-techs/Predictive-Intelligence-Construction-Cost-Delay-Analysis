import pandas as pd
import numpy as np

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing categorical values.
    For Equipment_Type, missing values are imputed as 'Unknown'.
    """
    df_clean = df.copy()
    if 'Equipment_Type' in df_clean.columns:
        df_clean['Equipment_Type'] = df_clean['Equipment_Type'].fillna('Unknown')
    return df_clean

def add_stage2_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-specific construction features for Stage 2 without leakage.
    - Progress_Gap: Planned_Progress_Percent - Actual_Progress_Percent
    - Labour_Deployment_Ratio: Actual_Labour_Count / Labour_Count
    - Labour_Shortage: Labour_Count - Actual_Labour_Count
    - Labour_Deployment_Gap: 1.0 - Labour_Deployment_Ratio
    - Material_Shortage: 100.0 - Material_Availability_Percent
    - Equipment_Shortage: 100.0 - Equipment_Availability_Percent
    - Progress_Ratio: Actual_Progress_Percent / Planned_Progress_Percent
    - Resource_Availability_Index: Average availability of material, equipment, and labour
    - Resource_Stress_Index: Composite metric of shortages and progress gap
    - Project_Size_Duration_Ratio: Area_sqft / Planned_Duration
    - Area_Per_Floor: Area_sqft / Number_of_Floors
    - Work_Rate_Deficit: Planned progress rate - Actual progress rate
    """
    df_feat = df.copy()
    
    # 1. Progress Gap
    if 'Planned_Progress_Percent' in df_feat.columns and 'Actual_Progress_Percent' in df_feat.columns:
        df_feat['Progress_Gap'] = df_feat['Planned_Progress_Percent'] - df_feat['Actual_Progress_Percent']
        
    # 2. Labour Deployment Ratio
    if 'Actual_Labour_Count' in df_feat.columns and 'Labour_Count' in df_feat.columns:
        df_feat['Labour_Deployment_Ratio'] = df_feat['Actual_Labour_Count'] / df_feat['Labour_Count'].replace(0, np.nan)
        df_feat['Labour_Deployment_Ratio'] = df_feat['Labour_Deployment_Ratio'].fillna(1.0)
        
    # 3. Labour Shortage
    if 'Actual_Labour_Count' in df_feat.columns and 'Labour_Count' in df_feat.columns:
        df_feat['Labour_Shortage'] = df_feat['Labour_Count'] - df_feat['Actual_Labour_Count']
        
    # 4. Labour Deployment Gap
    if 'Labour_Deployment_Ratio' in df_feat.columns:
        df_feat['Labour_Deployment_Gap'] = 1.0 - df_feat['Labour_Deployment_Ratio']
        
    # 5. Material Shortage
    if 'Material_Availability_Percent' in df_feat.columns:
        df_feat['Material_Shortage'] = 100.0 - df_feat['Material_Availability_Percent']
        
    # 6. Equipment Shortage
    if 'Equipment_Availability_Percent' in df_feat.columns:
        df_feat['Equipment_Shortage'] = 100.0 - df_feat['Equipment_Availability_Percent']
        
    # 7. Progress Ratio
    if 'Planned_Progress_Percent' in df_feat.columns and 'Actual_Progress_Percent' in df_feat.columns:
        df_feat['Progress_Ratio'] = df_feat['Actual_Progress_Percent'] / (df_feat['Planned_Progress_Percent'].replace(0, np.nan))
        df_feat['Progress_Ratio'] = df_feat['Progress_Ratio'].fillna(1.0)
        
    # 8. Resource Availability Index
    if 'Material_Availability_Percent' in df_feat.columns and 'Equipment_Availability_Percent' in df_feat.columns and 'Labour_Deployment_Ratio' in df_feat.columns:
        df_feat['Resource_Availability_Index'] = (df_feat['Material_Availability_Percent'] + df_feat['Equipment_Availability_Percent'] + np.clip(df_feat['Labour_Deployment_Ratio'], 0, 2) * 100.0) / 3.0
        
    # 9. Resource Stress Index
    if 'Material_Shortage' in df_feat.columns and 'Equipment_Shortage' in df_feat.columns and 'Labour_Deployment_Gap' in df_feat.columns and 'Progress_Gap' in df_feat.columns:
        df_feat['Resource_Stress_Index'] = (df_feat['Material_Shortage'] + df_feat['Equipment_Shortage'] + np.clip(df_feat['Labour_Deployment_Gap'], 0, 2) * 100.0 + np.maximum(0, df_feat['Progress_Gap'])) / 4.0
        
    # 10. Project Size Duration Ratio
    if 'Area_sqft' in df_feat.columns and 'Planned_Duration' in df_feat.columns:
        df_feat['Project_Size_Duration_Ratio'] = df_feat['Area_sqft'] / df_feat['Planned_Duration'].replace(0, np.nan)
        df_feat['Project_Size_Duration_Ratio'] = df_feat['Project_Size_Duration_Ratio'].fillna(0.0)

    # 11. Area Per Floor
    if 'Area_sqft' in df_feat.columns and 'Number_of_Floors' in df_feat.columns:
        df_feat['Area_Per_Floor'] = df_feat['Area_sqft'] / df_feat['Number_of_Floors'].replace(0, np.nan)
        df_feat['Area_Per_Floor'] = df_feat['Area_Per_Floor'].fillna(0.0)

    # 12. Work Rate Deficit, Rates, and Deficit Days
    if 'Planned_Progress_Percent' in df_feat.columns and 'Actual_Progress_Percent' in df_feat.columns and 'Planned_Duration' in df_feat.columns:
        planned_rate = df_feat['Planned_Progress_Percent'] / df_feat['Planned_Duration'].replace(0, np.nan)
        actual_rate = df_feat['Actual_Progress_Percent'] / df_feat['Planned_Duration'].replace(0, np.nan)
        df_feat['Work_Rate_Deficit'] = (planned_rate - actual_rate).fillna(0.0)
        df_feat['Planned_Work_Rate'] = planned_rate.fillna(0.0)
        df_feat['Actual_Work_Rate'] = actual_rate.fillna(0.0)

    if 'Progress_Gap' in df_feat.columns and 'Planned_Duration' in df_feat.columns:
        df_feat['Work_Deficit_Days'] = (df_feat['Progress_Gap'] / 100.0) * df_feat['Planned_Duration']
        df_feat['Work_Deficit_Days'] = df_feat['Work_Deficit_Days'].fillna(0.0)

    return df_feat 
