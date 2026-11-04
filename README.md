# Predictive Intelligence for Construction Cost Estimation and Project Delay Analysis

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask%203.1.3-green.svg)
![ML Stack](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20SHAP-orange.svg)
![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)
![License](https://img.shields.io/badge/License-Academic%20%2F%20Open-blue.svg)

---

## 📌 Executive Summary & Project Overview

The construction sector frequently suffers from massive budget overruns and unexpected schedule delays due to static estimation methods, unforeseen supply chain disruptions, and inadequate real-time monitoring. 

**"Predictive Intelligence for Construction Cost Estimation and Project Delay Analysis"** is an end-to-end Machine Learning and Explainable AI (XAI) decision-support system. It bridges the gap between pre-construction planning and runtime project execution through a two-stage predictive pipeline:

1. **Stage 1 (Pre-Construction Planning):** Estimates total project cost before breaking ground using structural specifications and baseline resource allocations.
2. **Stage 2 (Ongoing Construction Monitoring):** Tracks execution progress dynamically, predicting schedule delay days and multi-class delay risks (`Low`, `Medium`, `High`), backed by SHAP-driven root-cause explanations and corrective mitigation recommendations.

---

## 🎯 Problem Statement

Traditional construction project management relies heavily on static spreadsheets and subjective manual judgment. This leads to:
- **Cost Inaccuracies:** Inability to capture non-linear relationships among project area, multi-floor complexities, material tiers, and equipment requirements.
- **Blind Runtime Management:** Project delays are often recognized only after critical path disruptions become irreversible.
- **Black-Box Skepticism:** Traditional ML solutions output raw numbers without explaining *why* a budget is exceeded or *which* factor is causing the bottleneck.

---

## 💡 Key Objectives

- Develop a high-precision pre-construction cost estimation model trained on verified structural and budgetary attributes.
- Construct a dynamic ongoing monitoring module that evaluates project health using progress gaps and resource availability.
- Formulate a multi-class delay risk classifier to categorize potential slippage into actionable risk tiers (`Low`, `Medium`, `High`).
- Integrate **Explainable AI (SHAP)** into every inference stage to identify key cost/delay drivers transparently.
- Provide automated, context-aware mitigation recommendations for project managers to avert critical delays.
- Implement an intuitive, responsive web application powered by Flask and a modern dashboard UI.

---

## 🏗️ System Architecture & Workflow

```
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|             (Modern Glassmorphism Dashboard: HTML5, CSS3, JavaScript)             |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                               FLASK REST API LAYER                                |
|          (/api/stage1/predict, /api/stage2/predict, /api/projects, etc.)          |
+-----------------------------------------------------------------------------------+
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
+----------------------------------------+ +----------------------------------------+
|       STAGE 1: PRE-CONSTRUCTION        | |       STAGE 2: ONGOING MONITORING      |
|----------------------------------------| |----------------------------------------|
| • Preprocessing & Pipeline Encoders    | | • Domain Feature Engineering           |
| • Model: Tuned XGBoost Regressor       | |   (Progress Gap, Labour/Resource Index)|
| • Target: Total_Construction_Cost (INR)| | • Regressor: Extra Trees Regressor     |
| • Test R² = 0.9531 | MAE = ₹1.89 Cr    | |   (Expected Delay Days, R² = 0.7106)   |
| • SHAP TreeExplainer Cost Attribution  | | • Classifier: Random Forest Classifier |
| • Cost Optimization Recommendations    | |   (Low / Medium / High Risk, F1=0.8730)|
| • Persistent Project ID Storage        | | • Dynamic Risk Probability Breakdown   |
+----------------------------------------+ +----------------------------------------+
                     │                                       │
                     └───────────────────┬───────────────────┘
                                         ▼
+-----------------------------------------------------------------------------------+
|                        DATABASE & PERSISTENCE (SQLite3)                           |
|       (projects table, monitoring_records table, pre-seeded benchmarks)           |
+-----------------------------------------------------------------------------------+
```

---

## 🔬 Project Modules & Machine Learning Models

### 1. Stage 1 — Pre-Construction Cost Estimation
- **Objective:** Accurately forecast capital expenditure before execution commences.
- **Model:** **Tuned XGBoost Regressor**
- **Inputs:** `Project_Type`, `Location`, `Area_sqft`, `Number_of_Floors`, `Material_Type`, `Material_Cost`, `Labour_Cost`, `Labour_Count`, `Equipment_Type`, `Equipment_Cost`, `Planned_Duration`.
- **Performance:**
  - **Test $R^2$ Score:** `0.9531`
  - **Mean Absolute Error (MAE):** `₹18,988,712`
  - **Root Mean Squared Error (RMSE):** `₹32,614,496`
- **Explainability:** SHAP waterfall/summary plots quantifying how each design and resource choice impacts total expenditure.

### 2. Stage 2 — Ongoing Monitoring & Delay Analysis
- **Objective:** Detect deviations from planned milestones during active construction.
- **Delay Regression Model:** **Extra Trees Regressor**
  - **Target:** `Expected_Delay_Days`
  - **Test $R^2$ Score:** `0.7106`
  - **MAE:** `20.81 days` | **RMSE:** `43.34 days`
- **Delay Risk Classification Model:** **Random Forest Classifier (Class Weighted)**
  - **Target:** `Delay_Risk` (`Low`, `Medium`, `High`)
  - **Accuracy:** `87.33%`
  - **Weighted F1-Score:** `0.8730` | **Macro F1-Score:** `0.8368`
- **Domain-Engineered Metrics:**
  - `Progress_Gap = Planned_Progress_% - Actual_Progress_%`
  - `Labour_Ratio = Actual_Labour_Count / Planned_Labour_Count`
  - `Resource_Availability_Index = (Material_Availability + Equipment_Availability) / 2`
  - `Progress_Velocity = Actual_Progress_% / Elapsed_Time`

### 3. Explainable AI (XAI) & Recommendation Engine
- **TreeExplainer:** Computes exact additive feature attributions for every prediction without altering model inferences.
- **Adaptive Recommendations:** Dynamically generates prioritized mitigation strategies (e.g., labor reallocation, procurement expediting, shift adjustments) based on detected bottleneck severity.

---

## 📸 Application Screenshots

### 1. Stage 1 — Pre-Construction Cost Estimation
*Enter architectural specifications, material grades, and labor estimates to calculate estimated total project cost.*

![Stage 1 Pre-Construction Cost Estimation Input Form](screenshots/Screenshot%202026-10-05%20004126.png)

*Predicted total construction cost along with Explainable AI (SHAP) feature attributions identifying primary cost drivers.*

![Stage 1 Cost Estimation Prediction and SHAP Breakdown](screenshots/Screenshot%202026-10-05%20004141.png)

---

### 2. Stage 2 — Ongoing Construction Execution Monitoring
*Input live site milestones, actual vs. planned progress percentage, manpower deployment, and resource availability.*

![Stage 2 Ongoing Execution Monitoring Input Form](screenshots/Screenshot%202026-10-05%20004209.png)

*Forecasted schedule overrun (expected delay in days), multi-class risk classification, and probabilistic risk distribution.*

![Stage 2 Delay Overrun and Risk Probability Breakdown](screenshots/Screenshot%202026-10-05%20004236.png)

---

### 3. Explainable AI (XAI) & Actionable Recommendations
*SHAP root-cause feature impact breakdown indicating which parameters amplify or mitigate project delay risks.*

![Stage 2 SHAP Factors Influencing Delay Risk](screenshots/Screenshot%202026-10-05%20004256.png)

*Automated, prioritized mitigation strategies dynamically generated by the recommendation engine based on live site bottlenecks.*

![Context-Aware Delay Mitigation Recommendations](screenshots/Screenshot%202026-10-05%20004606.png)

---

### 4. Project Lifecycle & Progress Analytics
*End-to-end lifecycle tracking comparing Stage 1 baseline estimates with Stage 2 ongoing site monitoring snapshots.*

![Project Lifecycle and Milestone Tracking](screenshots/Screenshot%202026-10-05%20004704.png)

*Chronological planned vs. actual progress curves alongside structured historical audit snapshots.*

![Planned vs Actual Progress History and Snapshot Table](screenshots/Screenshot%202026-10-05%20010620.png)

---

## 💻 Tech Stack & Libraries

- **Programming Language:** Python 3.10+
- **Backend Framework:** Flask 3.1.3, Werkzeug 3.1.8, Jinja2 3.1.6, Flask-CORS 6.0.5
- **Machine Learning & Data Science:**
  - `scikit-learn 1.9.1`: Preprocessing pipelines, Random Forest, Extra Trees, cross-validation, and metrics
  - `xgboost 3.4.1`: Gradient-boosted decision trees for high-dimensional cost regression
  - `shap 0.52.0`: Model interpretability and Shapley value computations
  - `pandas 3.0.6` & `numpy 2.5.3`: High-performance data wrangling and numerical operations
  - `joblib 1.6.0`: Serialization for pre-trained pipelines and estimators
- **Frontend / UI:** HTML5, CSS3 (Modern Glassmorphism & Dashboard Layout), Vanilla JavaScript (Async Fetch API)
- **Database:** SQLite3 (`database/app_storage.db`) with migration handling and relational tracking

---

## 📂 Project Directory Structure

```text
PREDICTIVE_INTELLIGENCE/
│
├── app.py                     # Main Flask application entrypoint & REST API endpoints
├── requirements.txt           # Verified project dependencies
├── .gitignore                 # Standard Git exclusion file
├── README.md                  # Project documentation
│
├── src/                       # Core application modules & business logic
│   ├── database.py            # SQLite schema, CRUD operations & reference projects
│   ├── explainability.py      # SHAP explainer integration and plot generators
│   ├── feature_engineering.py # Data imputation & Stage 2 domain feature engineering
│   ├── model_training.py      # Model training routines, CV & performance evaluation
│   ├── prediction_service.py  # Unified Stage 1 & Stage 2 prediction interface
│   ├── preprocessing.py       # ColumnTransformer & categorical encoding pipelines
│   └── recommendation_engine.py # Context-aware delay and cost mitigation rules
│
├── models/                    # Trained model binaries and reproducible code
│   ├── stage1_cost_model.joblib            # Trained XGBoost Cost model
│   ├── stage1_cost_model.py                # Stage 1 training script
│   ├── stage1_pipeline.joblib              # Stage 1 ColumnTransformer pipeline
│   ├── stage2_delay_model.joblib           # Trained Extra Trees Delay model
│   ├── stage2_delay_model.py               # Stage 2 Delay training script
│   ├── stage2_pipeline.joblib              # Stage 2 ColumnTransformer pipeline
│   ├── stage2_risk_model.joblib            # Trained Random Forest Classifier
│   ├── stage2_risk_model.py                # Stage 2 Risk training script
│   └── stage2_risk_label_encoder.joblib    # Multi-class label encoder (Low/Med/High)
│
├── dataset/                   # Verified data source
│   └── construction_cost_delay_dataset_clean.csv # Cleaned construction dataset
│
├── templates/                 # UI HTML templates
│   └── index.html             # Single-page dashboard interface
│
├── static/                    # Frontend static assets
│   ├── css/
│   │   └── style.css          # Responsive styling and glassmorphism themes
│   └── js/
│       └── app.js             # Client-side API interactions & UI updates
│
├── outputs/                   # Performance reports and visualizations
│   ├── feature_summary.csv
│   ├── final_validation_report.txt
│   ├── preprocessing_summary.csv
│   ├── recommendation_results.csv
│   ├── stage1_model_comparison.csv
│   ├── stage2_delay_model_comparison.csv
│   ├── stage2_risk_model_comparison.csv
│   └── shap/                  # Generated SHAP summary and importance plots
│
├── tests/                     # Validation test suites
│   ├── test_backend.py        # Comprehensive backend and endpoint testing
│   └── test_stage2_shap.py    # SHAP interpretability verification
│
├── screenshots/               # Application UI and execution screenshots
│   ├── Screenshot 2026-10-05 004126.png
│   ├── Screenshot 2026-10-05 004141.png
│   ├── Screenshot 2026-10-05 004209.png
│   ├── Screenshot 2026-10-05 004236.png
│   ├── Screenshot 2026-10-05 004256.png
│   ├── Screenshot 2026-10-05 004606.png
│   ├── Screenshot 2026-10-05 004704.png
│   └── Screenshot 2026-10-05 010620.png
│
└── database/                  # Local SQLite storage
    └── app_storage.db         # Persistent storage for projects & monitoring snapshots
```

---

## ⚙️ Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed
- Git installed on your system
- Modern web browser (Chrome, Edge, Firefox, Brave)

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd PREDICTIVE_INTELLIGENCE
```

### 2. Create and Activate a Virtual Environment
- **Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project Locally

### 1. Start the Flask Server
```bash
python app.py
```
Upon startup, the console will confirm that:
- SQLite tables are initialized and seeded.
- Pre-trained ML models, pipelines, and SHAP explainers are loaded.
- The web server is running at:
  ```
  ==================================================
  Starting Predictive Intelligence Backend API Server
  URL: http://127.0.0.1:5000
  ==================================================
  ```

### 2. Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### 3. Running Automated Tests
To run the automated verification and backend validation tests:
```bash
python tests/test_backend.py
python tests/test_stage2_shap.py
```

---

## 📊 Application REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the web dashboard UI |
| `GET` | `/api/health` | Service health status and model availability |
| `POST` | `/api/stage1/predict` | Computes pre-construction cost estimation and SHAP factors |
| `GET` | `/api/projects` | Lists all stored project records |
| `GET` | `/api/projects/<project_id>` | Retrieves full details for a specific project |
| `POST` | `/api/stage2/predict` | Analyzes ongoing progress, delay days, and risk classification |
| `GET` | `/api/projects/<project_id>/monitoring` | Returns chronological snapshot history for a project |
| `GET` | `/api/model-performance` | Returns benchmark performance metrics for all models |
| `POST` | `/api/projects/<project_id>/complete` | Updates project lifecycle completion status |

---

## 🛡️ Data Leakage & Security Measures

- **Zero Information Leakage:** Downstream execution variables (actual delay, actual costs, completion status) are strictly excluded from Stage 1 pre-construction feature sets.
- **Isolated Target Variables:** In Stage 2, output metrics (`Expected_Delay_Days` and `Delay_Risk`) are never passed as prediction features.
- **Safety & Secret Isolation:** No private credentials, database passwords, or hard-coded API secrets are stored within the codebase.

---

## 🎓 Academic Attribution

- **Project:** Master of Computer Applications (MCA) Final Project
- **Topic:** Predictive Intelligence for Construction Cost Estimation and Project Delay Analysis
- **Domain:** Artificial Intelligence / Machine Learning in Civil & Project Management
