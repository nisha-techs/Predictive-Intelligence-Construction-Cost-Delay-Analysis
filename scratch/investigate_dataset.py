import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.feature_engineering import add_stage2_engineered_features, impute_missing_values
from src.preprocessing import get_feature_definitions

df_raw = pd.read_csv("dataset/construction_cost_delay_dataset_clean.csv")
print("=== DATASET COLUMNS ===")
print(df_raw.columns.tolist())

print("\n=== HEAD OF DATASET ===")
print(df_raw[['Actual_Progress_Percent', 'Actual_Labour_Count', 'Material_Availability_Percent', 'Equipment_Availability_Percent', 'Expected_Delay_Days', 'Delay_Risk']].head(10))

print("\n=== SUMMARY STATS BY DELAY_RISK IN CLEAN DATASET ===")
print(df_raw.groupby('Delay_Risk')[['Actual_Progress_Percent', 'Actual_Labour_Count', 'Labour_Count', 'Material_Availability_Percent', 'Equipment_Availability_Percent', 'Expected_Delay_Days']].mean())

print("\n=== CHECKING HOW DELAY_RISK IS DISTRIBUTED RELATIVE TO EXPECTED_DELAY_DAYS ===")
print(df_raw.groupby('Delay_Risk')['Expected_Delay_Days'].describe())

print("\n=== CHECKING HOW DELAY_RISK IS DISTRIBUTED RELATIVE TO ACTUAL_PROGRESS_PERCENT ===")
print(df_raw.groupby('Delay_Risk')['Actual_Progress_Percent'].describe())

print("\n=== CHECKING HOW DELAY_RISK IS DISTRIBUTED RELATIVE TO MATERIAL_AVAILABILITY_PERCENT ===")
print(df_raw.groupby('Delay_Risk')['Material_Availability_Percent'].describe())

print("\n=== CHECKING HOW DELAY_RISK IS DISTRIBUTED RELATIVE TO EQUIPMENT_AVAILABILITY_PERCENT ===")
print(df_raw.groupby('Delay_Risk')['Equipment_Availability_Percent'].describe())
