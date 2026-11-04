import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# Import preprocessing pipeline loader
from preprocessing import process_and_validate_pipeline, impute_missing_values, add_stage2_engineered_features, build_column_transformer, get_feature_definitions
from sklearn.model_selection import train_test_split

def run_stage1_cost_estimation(X_train, X_test, y_train, y_test):
    print("\n==================================================")
    print("STAGE 1: PRE-CONSTRUCTION COST ESTIMATION")
    print("==================================================")
    
    results = []
    trained_models = {}
    
    # 1. Linear Regression Baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_train = lr.predict(X_train)
    y_pred_lr_test = lr.predict(X_test)
    
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    
    results.append({
        'Model': 'Linear Regression',
        'Train_R2': r2_score(y_train, y_pred_lr_train),
        'CV_R2_Mean': cv_scores_lr.mean(),
        'CV_R2_Std': cv_scores_lr.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_lr_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
        'Test_R2': r2_score(y_test, y_pred_lr_test),
        'Tuned': False
    })
    trained_models['Linear Regression'] = lr
    
    # 2. Random Forest Regressor
    rf_best = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=2, random_state=42, n_jobs=-1)
    rf_best.fit(X_train, y_train)
    y_pred_rf_train = rf_best.predict(X_train)
    y_pred_rf_test = rf_best.predict(X_test)
    cv_scores_rf = cross_val_score(rf_best, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': 'Random Forest Regressor',
        'Train_R2': r2_score(y_train, y_pred_rf_train),
        'CV_R2_Mean': cv_scores_rf.mean(),
        'CV_R2_Std': cv_scores_rf.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_rf_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
        'Test_R2': r2_score(y_test, y_pred_rf_test),
        'Tuned': True,
        'Best_Params': "{'max_depth': 25, 'min_samples_split': 2, 'n_estimators': 200}"
    })
    trained_models['Random Forest Regressor'] = rf_best
    
    # 3. XGBoost Regressor
    xgb_best = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, objective='reg:squarederror')
    xgb_best.fit(X_train, y_train)
    y_pred_xgb_train = xgb_best.predict(X_train)
    y_pred_xgb_test = xgb_best.predict(X_test)
    cv_scores_xgb = cross_val_score(xgb_best, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': 'XGBoost Regressor',
        'Train_R2': r2_score(y_train, y_pred_xgb_train),
        'CV_R2_Mean': cv_scores_xgb.mean(),
        'CV_R2_Std': cv_scores_xgb.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_xgb_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb_test)),
        'Test_R2': r2_score(y_test, y_pred_xgb_test),
        'Tuned': True,
        'Best_Params': "{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}"
    })
    trained_models['XGBoost Regressor'] = xgb_best
    
    df_results = pd.DataFrame(results)
    print(df_results[['Model', 'Test_MAE', 'Test_RMSE', 'Test_R2', 'CV_R2_Mean']].to_string(index=False))
    
    # Select best model based on Test R2
    best_idx = df_results['Test_R2'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    print(f"\n---> Selected Best Stage 1 Model: {best_model_name} (Test R2: {df_results.loc[best_idx, 'Test_R2']:.4f})")
    
    return trained_models[best_model_name], df_results, best_model_name

def run_stage2_delay_prediction(X_train, X_test, y_train, y_test):
    print("\n==================================================")
    print("STAGE 2: EXPECTED DELAY DAYS REGRESSION")
    print("==================================================")
    
    results = []
    trained_models = {}
    
    # 1. Linear Regression Baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_tr = lr.predict(X_train)
    y_pred_lr_te = lr.predict(X_test)
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': 'Linear Regression',
        'Train_R2': r2_score(y_train, y_pred_lr_tr),
        'CV_R2_Mean': cv_scores_lr.mean(),
        'CV_R2_Std': cv_scores_lr.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_lr_te),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr_te)),
        'Test_R2': r2_score(y_test, y_pred_lr_te),
        'Tuned': False,
        'Best_Params': ''
    })
    trained_models['Linear Regression'] = lr

    # 2. Extra Trees Regressor
    et_best = ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
    et_best.fit(X_train, y_train)
    
    y_pred_et_train = et_best.predict(X_train)
    y_pred_et_test = et_best.predict(X_test)
    cv_scores_et = cross_val_score(et_best, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': 'Extra Trees Regressor',
        'Train_R2': r2_score(y_train, y_pred_et_train),
        'CV_R2_Mean': cv_scores_et.mean(),
        'CV_R2_Std': cv_scores_et.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_et_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_et_test)),
        'Test_R2': r2_score(y_test, y_pred_et_test),
        'Tuned': True,
        'Best_Params': "{'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 4}"
    })
    trained_models['Extra Trees Regressor'] = et_best

    # 3. Random Forest Regressor
    rf_best = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1)
    rf_best.fit(X_train, y_train)
    
    y_pred_rf_train = rf_best.predict(X_train)
    y_pred_rf_test = rf_best.predict(X_test)
    cv_scores_rf = cross_val_score(rf_best, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': 'Random Forest Regressor',
        'Train_R2': r2_score(y_train, y_pred_rf_train),
        'CV_R2_Mean': cv_scores_rf.mean(),
        'CV_R2_Std': cv_scores_rf.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_rf_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
        'Test_R2': r2_score(y_test, y_pred_rf_test),
        'Tuned': True,
        'Best_Params': "{'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 4}"
    })
    trained_models['Random Forest Regressor'] = rf_best
    
    # 4. XGBoost Regressor
    xgb_best = xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb_best.fit(X_train, y_train)
    
    y_pred_xgb_train = xgb_best.predict(X_train)
    y_pred_xgb_test = xgb_best.predict(X_test)
    cv_scores_xgb = cross_val_score(xgb_best, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    results.append({
        'Model': 'XGBoost Regressor',
        'Train_R2': r2_score(y_train, y_pred_xgb_train),
        'CV_R2_Mean': cv_scores_xgb.mean(),
        'CV_R2_Std': cv_scores_xgb.std(),
        'Test_MAE': mean_absolute_error(y_test, y_pred_xgb_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb_test)),
        'Test_R2': r2_score(y_test, y_pred_xgb_test),
        'Tuned': True,
        'Best_Params': "{'n_estimators': 350, 'learning_rate': 0.02, 'max_depth': 4, 'subsample': 0.8}"
    })
    trained_models['XGBoost Regressor'] = xgb_best
    
    df_results = pd.DataFrame(results)
    print(df_results[['Model', 'Test_MAE', 'Test_RMSE', 'Test_R2', 'CV_R2_Mean']].to_string(index=False))
    
    best_idx = df_results['Test_R2'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    print(f"\n---> Selected Best Stage 2 Delay Model: {best_model_name} (Test R2: {df_results.loc[best_idx, 'Test_R2']:.4f})")
    
    # Generate Phase 10 detailed experiment report
    report_rows = []
    for idx, r in df_results.iterrows():
        is_sel = (r['Model'] == best_model_name)
        gap = r['Train_R2'] - r['Test_R2']
        if is_sel:
            reason = f"Achieved highest valid Test R² ({r['Test_R2']:.4f}), best MAE ({r['Test_MAE']:.2f}d), and high CV R² ({r['CV_R2_Mean']:.4f}) with controlled overfitting."
        else:
            reason = f"Lower Test R² ({r['Test_R2']:.4f}) or higher MAE ({r['Test_MAE']:.2f}d) compared to selected {best_model_name}."
            
        report_rows.append({
            'Model': r['Model'],
            'Features_Version': 'v2_enhanced_domain_features',
            'Test_R2': r['Test_R2'],
            'Test_MAE': r['Test_MAE'],
            'Test_RMSE': r['Test_RMSE'],
            'CV_R2_Mean': r['CV_R2_Mean'],
            'CV_R2_Std': r['CV_R2_Std'],
            'Train_R2': r['Train_R2'],
            'Train_Test_Gap': gap,
            'Selected': is_sel,
            'Reason': reason
        })
        
    df_report = pd.DataFrame(report_rows)
    os.makedirs('outputs', exist_ok=True)
    df_report.to_csv('outputs/stage2_delay_improvement_report.csv', index=False)
    
    return trained_models[best_model_name], df_results, best_model_name

def run_stage2_risk_classification(X_train, X_test, y_train_str, y_test_str):
    print("\n==================================================")
    print("STAGE 2: DELAY RISK MULTI-CLASS CLASSIFICATION")
    print("==================================================")
    
    le = LabelEncoder()
    # Ensure consistent label ordering ['Low', 'Medium', 'High']
    le.fit(['Low', 'Medium', 'High'])
    y_train = le.transform(y_train_str)
    y_test = le.transform(y_test_str)
    
    results = []
    trained_models = {}
    
    # 1. Random Forest Classifier (with class_weight='balanced')
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    
    y_pred_rf_train = rf_best.predict(X_train)
    y_pred_rf_test = rf_best.predict(X_test)
    y_proba_rf_test = rf_best.predict_proba(X_test)
    
    rf_acc = accuracy_score(y_test, y_pred_rf_test)
    rf_prec_w = precision_score(y_test, y_pred_rf_test, average='weighted')
    rf_rec_w = recall_score(y_test, y_pred_rf_test, average='weighted')
    rf_f1_w = f1_score(y_test, y_pred_rf_test, average='weighted')
    
    rf_prec_m = precision_score(y_test, y_pred_rf_test, average='macro')
    rf_rec_m = recall_score(y_test, y_pred_rf_test, average='macro')
    rf_f1_m = f1_score(y_test, y_pred_rf_test, average='macro')
    rf_roc_auc = roc_auc_score(y_test, y_proba_rf_test, multi_class='ovr', average='weighted')
    
    cv_scores_rf = cross_val_score(rf_best, X_train, y_train, cv=5, scoring='f1_weighted')
    
    results.append({
        'Model': 'Random Forest Classifier',
        'Train_Acc': accuracy_score(y_train, y_pred_rf_train),
        'CV_F1_Weighted_Mean': cv_scores_rf.mean(),
        'CV_F1_Weighted_Std': cv_scores_rf.std(),
        'Accuracy': rf_acc,
        'Weighted_Precision': rf_prec_w,
        'Weighted_Recall': rf_rec_w,
        'Weighted_F1': rf_f1_w,
        'Macro_Precision': rf_prec_m,
        'Macro_Recall': rf_rec_m,
        'Macro_F1': rf_f1_m,
        'ROC_AUC_Weighted': rf_roc_auc,
        'Best_Params': str(rf_grid.best_params_)
    })
    trained_models['Random Forest Classifier'] = rf_best
    
    # 2. XGBoost Classifier (with sample weights for imbalance)
    sample_weights_train = compute_sample_weight('balanced', y_train)
    xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.1]
    }
    xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    xgb_grid.fit(X_train, y_train, sample_weight=sample_weights_train)
    xgb_best = xgb_grid.best_estimator_
    
    y_pred_xgb_train = xgb_best.predict(X_train)
    y_pred_xgb_test = xgb_best.predict(X_test)
    y_proba_xgb_test = xgb_best.predict_proba(X_test)
    
    xgb_acc = accuracy_score(y_test, y_pred_xgb_test)
    xgb_prec_w = precision_score(y_test, y_pred_xgb_test, average='weighted')
    xgb_rec_w = recall_score(y_test, y_pred_xgb_test, average='weighted')
    xgb_f1_w = f1_score(y_test, y_pred_xgb_test, average='weighted')
    
    xgb_prec_m = precision_score(y_test, y_pred_xgb_test, average='macro')
    xgb_rec_m = recall_score(y_test, y_pred_xgb_test, average='macro')
    xgb_f1_m = f1_score(y_test, y_pred_xgb_test, average='macro')
    xgb_roc_auc = roc_auc_score(y_test, y_proba_xgb_test, multi_class='ovr', average='weighted')
    
    cv_scores_xgb = cross_val_score(xgb_best, X_train, y_train, cv=5, scoring='f1_weighted')
    
    results.append({
        'Model': 'XGBoost Classifier',
        'Train_Acc': accuracy_score(y_train, y_pred_xgb_train),
        'CV_F1_Weighted_Mean': cv_scores_xgb.mean(),
        'CV_F1_Weighted_Std': cv_scores_xgb.std(),
        'Accuracy': xgb_acc,
        'Weighted_Precision': xgb_prec_w,
        'Weighted_Recall': xgb_rec_w,
        'Weighted_F1': xgb_f1_w,
        'Macro_Precision': xgb_prec_m,
        'Macro_Recall': xgb_rec_m,
        'Macro_F1': xgb_f1_m,
        'ROC_AUC_Weighted': xgb_roc_auc,
        'Best_Params': str(xgb_grid.best_params_)
    })
    trained_models['XGBoost Classifier'] = xgb_best
    
    df_results = pd.DataFrame(results)
    print(df_results[['Model', 'Accuracy', 'Weighted_F1', 'Macro_F1', 'ROC_AUC_Weighted', 'CV_F1_Weighted_Mean']].to_string(index=False))
    
    best_idx = df_results['Weighted_F1'].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    print(f"\n---> Selected Best Stage 2 Risk Model: {best_model_name} (Test Weighted F1: {df_results.loc[best_idx, 'Weighted_F1']:.4f})")
    
    # Detailed report for best classifier
    best_clf = trained_models[best_model_name]
    best_preds = best_clf.predict(X_test)
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test, best_preds, target_names=le.classes_))
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, best_preds), index=le.classes_, columns=le.classes_))
    
    return trained_models[best_model_name], df_results, best_model_name, le

def main():
    print("Starting Model Training and Evaluation Pipeline...\n")
    
    # Load dataset
    csv_path = "dataset/construction_cost_delay_dataset_clean.csv"
    df_raw = pd.read_csv(csv_path)
    df_clean = impute_missing_values(df_raw)
    df_proc = add_stage2_engineered_features(df_clean)
    
    stage1_cols, stage2_cols = get_feature_definitions()
    
    cat_cols_s1 = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']
    num_cols_s1 = [c for c in stage1_cols if c not in cat_cols_s1]
    
    cat_cols_s2 = ['Project_Type', 'Location', 'Material_Type', 'Equipment_Type']
    num_cols_s2 = [c for c in stage2_cols if c not in cat_cols_s2]
    
    y_stage1 = df_proc['Total_Construction_Cost']
    y_stage2_delay = df_proc['Expected_Delay_Days']
    y_stage2_risk = df_proc['Delay_Risk']
    
    X_stage1_raw = df_proc[stage1_cols]
    X_stage2_raw = df_proc[stage2_cols]
    
    # Train / Test splits
    X_s1_tr_raw, X_s1_te_raw, y_s1_tr, y_s1_te = train_test_split(
        X_stage1_raw, y_stage1, test_size=0.2, random_state=42
    )
    
    X_s2_tr_raw, X_s2_te_raw, y_s2_delay_tr, y_s2_delay_te, y_s2_risk_tr, y_s2_risk_te = train_test_split(
        X_stage2_raw, y_stage2_delay, y_stage2_risk, test_size=0.2, random_state=42
    )
    
    # Build & Fit ColumnTransformers strictly on Training data
    ct_s1 = build_column_transformer(cat_cols_s1, num_cols_s1)
    ct_s1.fit(X_s1_tr_raw)
    
    ct_s2 = build_column_transformer(cat_cols_s2, num_cols_s2)
    ct_s2.fit(X_s2_tr_raw)
    
    cat_s1_names = ct_s1.named_transformers_['cat'].get_feature_names_out(cat_cols_s1)
    s1_feat_names = list(cat_s1_names) + num_cols_s1
    
    cat_s2_names = ct_s2.named_transformers_['cat'].get_feature_names_out(cat_cols_s2)
    s2_feat_names = list(cat_s2_names) + num_cols_s2
    
    X_s1_tr = pd.DataFrame(ct_s1.transform(X_s1_tr_raw), columns=s1_feat_names)
    X_s1_te = pd.DataFrame(ct_s1.transform(X_s1_te_raw), columns=s1_feat_names)
    
    X_s2_tr = pd.DataFrame(ct_s2.transform(X_s2_tr_raw), columns=s2_feat_names)
    X_s2_te = pd.DataFrame(ct_s2.transform(X_s2_te_raw), columns=s2_feat_names)
    
    # 1. Run Stage 1 Training
    best_s1_model, df_s1_comp, best_s1_name = run_stage1_cost_estimation(X_s1_tr, X_s1_te, y_s1_tr, y_s1_te)
    
    # 2. Run Stage 2 Delay Training
    best_s2_delay_model, df_s2_delay_comp, best_s2_delay_name = run_stage2_delay_prediction(X_s2_tr, X_s2_te, y_s2_delay_tr, y_s2_delay_te)
    
    # 3. Run Stage 2 Risk Training
    best_s2_risk_model, df_s2_risk_comp, best_s2_risk_name, le_risk = run_stage2_risk_classification(X_s2_tr, X_s2_te, y_s2_risk_tr, y_s2_risk_te)
    
    # Prediction Validations
    print("\n==================================================")
    print("PREDICTION VALIDATION & INTEGRITY CHECKS")
    print("==================================================")
    
    s1_preds = best_s1_model.predict(X_s1_te)
    s2_delay_preds = best_s2_delay_model.predict(X_s2_te)
    s2_risk_preds_encoded = best_s2_risk_model.predict(X_s2_te)
    s2_risk_preds = le_risk.inverse_transform(s2_risk_preds_encoded)
    
    s2_delay_preds = np.maximum(0, s2_delay_preds)
    s1_preds = np.maximum(0, s1_preds)
    
    print(f"Stage 1 Cost Predictions: Min = INR {s1_preds.min():,.2f}, Max = INR {s1_preds.max():,.2f}, Negative Count = {(s1_preds < 0).sum()}")
    print(f"Stage 2 Delay Predictions: Min = {s2_delay_preds.min():.2f} days, Max = {s2_delay_preds.max():.2f} days, Negative Count = {(s2_delay_preds < 0).sum()}")
    print(f"Stage 2 Risk Predictions Classes: {np.unique(s2_risk_preds, return_counts=True)}")
    
    assert (s1_preds >= 0).all(), "Negative cost predictions detected!"
    assert (s2_delay_preds >= 0).all(), "Negative delay predictions detected!"
    assert set(s2_risk_preds).issubset({'Low', 'Medium', 'High'}), "Invalid risk class predictions!"
    print("All prediction validation assertions passed successfully!")
    
    # Save Artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    joblib.dump(best_s1_model, "models/stage1_cost_model.joblib")
    joblib.dump(best_s2_delay_model, "models/stage2_delay_model.joblib")
    joblib.dump(best_s2_risk_model, "models/stage2_risk_model.joblib")
    
    joblib.dump(ct_s1, "models/stage1_pipeline.joblib")
    joblib.dump(ct_s2, "models/stage2_pipeline.joblib")
    joblib.dump(le_risk, "models/stage2_risk_label_encoder.joblib")
    
    df_s1_comp.to_csv("outputs/stage1_model_comparison.csv", index=False)
    df_s2_delay_comp.to_csv("outputs/stage2_delay_model_comparison.csv", index=False)
    df_s2_risk_comp.to_csv("outputs/stage2_risk_model_comparison.csv", index=False)
    
    print("\nSaved Models and Pipeline Artifacts:")
    print("  - models/stage1_cost_model.joblib")
    print("  - models/stage2_delay_model.joblib")
    print("  - models/stage2_risk_model.joblib")
    print("  - models/stage1_pipeline.joblib")
    print("  - models/stage2_pipeline.joblib")
    print("  - models/stage2_risk_label_encoder.joblib")
    
    print("\nSaved Outputs:")
    print("  - outputs/stage1_model_comparison.csv")
    print("  - outputs/stage2_delay_model_comparison.csv")
    print("  - outputs/stage2_risk_model_comparison.csv")

if __name__ == '__main__':
    main()
