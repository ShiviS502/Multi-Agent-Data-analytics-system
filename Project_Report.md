# Project Report: Agentic AI Data Analysis System

## Executive Overview
The Agentic AI Data Analysis System is an automated, multi-agent machine learning pipeline built using LangGraph, LangChain, roq LLM(open/gpt-oss-120b), and moden Python data science libraries(pandas, scikit-learn).
Unlike traditional static scripts, this framework decomposes data science tasks into speacialized autonomous agents. Each agent possesses a single responsibility[planing, data ingestion, data preperation, AutoML model training, and executive reporting] communicating through a shared state machine to complete end-to-end data tasks with minimal human intervention.

## Project Architechture & Agent Workflow
The system is organized around a stateful graph where each node represents an isolated agent that reads from and writes to a shared state(AgentState).
```plaintext
[Start] 
   │
   ▼
[🤖 Planner Agent] ──► Generates structured project plan & identifies target strategy
   │
   ▼
[🔍 Data Finder Agent] ──► Ingests/generates raw dataset based on plan
   │
   ▼
[🧹 Data Prep Agent] ──► Handles missing values, scaling, and categorical encoding
   │
   ▼
[📊 Modeling Agent] ──► Trains ML models (Classification/Regression) & computes metrics
   │
   ▼
[📝 Reporter Agent] ──► Synthesizes all downstream outputs into an executive report
   │
   ▼
 [End]
```
 ### Specialized Agents
 1. **Planner Agent:** Receives high-level user intent (e.g., "Find heart disease data, preprocess it, and build a predictive model"), queries the LLM, and produces a structured plan defininf the ML task type (classification/ regression), target column, and preprocessing requirements.
 2. **Data Finder Agent:** Ingests external datasets or generates synthetic datasets tailored to the plan's specification, outputting raw data structures alongside missinness statistics
 3. **Data Perp & Cleaner Agent:** Applies preprocessing pipelines using SimpleImputer, StandardScaler, and OneHotEncoder to prepare clean numeric feature sets ready for algorithm training.
 4. **AutoML & Modeling Agent:** Evaluates task requirements, performs automated train-test splitting, fits ensemble models (RandomForestClassifier or RandomForestRegressor), and records standard metrics ($Accuracy$, $F_1\text{-score}$, $RMSE$, $R^2$)
 5. **Reporter Agent:** Aggregates findings across all states and prompts the LLM to forat a structured executive summay complete with technical tables and actionable reccomendations.

## Tech Stack & Dependencies
- Orchestration: langgraph, langchain-core
- LLM Engine: langchain-groq (openai/gpt-oss-120b)
- Data Processing & ML: pandas, numpy, scikit-learn
- Evironment & Package anagement: uv
- Version Control: git

## Installation & Setup Guide
1. Prerequisites
   Ensure Python 3.10+ and uv are installed on your machine.
2. Environment Setup
   Clone the repository and install project dependencies using uv:
   ```python
   cd "path/to/Agentic AI data analysis system"
   uv pip install langchain-groq langgraph pandas numpy scikit-learn
   ```
3. API Key Configuration
   Obtain a free API key from console.groq.com and set it in your environment:
   ```python
   $env:GROQ_API_KEY="gsk_your_groq_api_key_here"
   export GROQ_API_KEY="gsk_your_groq_api_key_here"
   ```
4. Running the Pipeline
   Execute the main entry script:
   ```python
   uv run python main.py
   ```
## Execution & Test Results
Actual Terminal Execution Log
```plaintext
uv run python main.py

[🤖 Planner Agent] Analyzing goal and structuring execution plan using Groq...
[🤖 Planner Agent] Target Column: 'target' | Task Type: 'classification'

[🔍 Data Finder Agent] Fetching/generating dataset...
[🔍 Data Finder Agent] Dataset prepared at './data/sample_data.csv' with shape [200, 5]

[🧹 Data Prep Agent] Cleaning, imputing, and scaling dataset...
C:\Users\Ram&Krishna\Projects\Agentic AI data analysis system\main.py:128: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
[🧹 Data Prep Agent] Cleaned dataset saved to './data/cleaned_dataset.csv'. Processed 3 numeric & 1 categorical features.

[📊 Modeling Agent] Training model and calculating metrics...
[📊 Modeling Agent] Model Trained: RandomForest | Metrics: {'accuracy': 0.425, 'f1_score': 0.4203}

[📝 Reporter Agent] Compiling final summary report using Groq...

==================================================
FINAL EXECUTIVE REPORT
==================================================
**Executive Summary**  
The project aimed to locate a heart‑disease dataset, prepare it for analysis, and develop a classification model to predict disease risk. A publicly‑available UCI‑style dataset (200 records) was acquired, cleaned, and expanded to include an additional engineered feature, resulting in a final table of 200 × 6. A Random Forest classifier was trained, achieving **42.5 % accuracy** and an **F1‑score of 0.420**—well below acceptable clinical‑decision thresholds, indicating that the current data and modeling approach are insufficient for reliable risk prediction.

---

**Data Preprocessing Actions**  

| Step | Action | Rationale |
|------|--------|-----------|
| Missing‑value handling | Imputed or removed rows with nulls | Prevents model bias and training errors |
| Categorical encoding | One‑hot/ordinal encoding for gender, chest‑pain type, etc. | Enables tree‑based algorithms to interpret non‑numeric data |
| Scaling/normalization | Applied Min‑Max scaling to numeric features | Improves convergence for algorithms that are distance‑sensitive (future model extensions) |
| Class balancing | Checked class distribution; no oversampling/undersampling applied (minor imbalance) | Maintains original prevalence while avoiding synthetic bias |
| Train/validation/test split | 70 % / 15 % / 15 % stratified split | Ensures unbiased performance estimates |
| Feature engineering | Added BMI (weight/height²) and age bins | Attempts to capture non‑linear risk patterns |

The cleaned dataset grew from 5 to 6 columns after adding engineered features.

---

**Model Performance**  

- **Algorithm:** Random Forest (default hyper‑parameters)  
- **Accuracy:** 0.425  
- **F1‑Score:** 0.4203  

These metrics suggest the model is only marginally better than random guessing for a binary outcome, reflecting limited predictive power.

---

**Recommendations**  

1. **Data Enrichment**  
   - Acquire a larger, more representative heart‑disease dataset (e.g., the full UCI Cleveland, Statlog, or a recent Kaggle collection) to increase sample size and feature diversity.  
   - Incorporate additional clinically relevant variables (cholesterol, blood pressure, smoking status, family history, ECG results).

2. **Advanced Feature Engineering**  
   - Create interaction terms (e.g., age × cholesterol) and non‑linear transformations (log, polynomial).  
   - Use domain‑driven risk scores (e.g., Framingham risk factors) as composite features.

3. **Model Tuning & Alternatives**  
   - Perform hyper‑parameter optimization (grid/random search, Bayesian optimization) for tree depth, number of estimators, class weighting.  
   - Experiment with gradient‑boosting models (XGBoost, LightGBM) and logistic regression with regularization.  
   - Evaluate ensemble stacking to combine strengths of multiple classifiers.

4. **Address Class Imbalance**  
   - If the positive class is under‑represented, apply SMOTE or class‑weight adjustments to improve recall and F1.

5. **Robust Validation**  
   - Use stratified k‑fold cross‑validation (k = 5–10) to obtain stable performance estimates.  
   - Report additional metrics (ROC‑AUC, sensitivity, specificity) to assess clinical utility.

6. **Pipeline Automation**  
   - Implement the preprocessing and modeling steps in a reproducible pipeline (e.g., scikit‑learn `Pipeline` or MLflow) to streamline experimentation and ensure traceability.

By expanding the dataset, enriching features, and rigorously tuning the model, predictive performance can be expected to rise substantially, moving toward a clinically useful heart‑disease risk tool.
PS C:\Users\Ram&Krishna\Projects\Agentic AI data analysis system> uv run python main.py

[🤖 Planner Agent] Analyzing goal and structuring execution plan using Groq...
[🤖 Planner Agent] Target Column: 'target' | Task Type: 'classification'

[🔍 Data Finder Agent] Fetching/generating dataset...
[🔍 Data Finder Agent] Dataset prepared at './data/sample_data.csv' with shape [200, 5]

[🧹 Data Prep Agent] Cleaning, imputing, and scaling dataset...
[🧹 Data Prep Agent] Cleaned dataset saved to './data/cleaned_dataset.csv'. Processed 3 numeric & 1 categorical features.

[📊 Modeling Agent] Training model and calculating metrics...
[📊 Modeling Agent] Model Trained: RandomForest | Metrics: {'accuracy': 0.425, 'f1_score': 0.4203}

[📝 Reporter Agent] Compiling final summary report using Groq...

==================================================
FINAL EXECUTIVE REPORT
==================================================
**Executive Summary**  
The project aimed to locate a heart‑disease dataset, prepare it for analysis, and develop a classification model to predict disease risk. A publicly‑available UCI‑style dataset (200 records) was retrieved, cleaned, and expanded to include an additional engineered feature, resulting in a final shape of **200 × 6**. A Random Forest classifier was trained, achieving **accuracy = 42.5 %** and **F1‑score = 0.420**—well below acceptable clinical‑decision thresholds, indicating that the current data and modeling approach are insufficient for reliable risk prediction.

---

**Data Preprocessing Actions**  

| Step | Action | Rationale |
|------|--------|-----------|
| **Missing‑value handling** | Imputed or removed rows with NaNs | Prevents model bias and training failures |
| **Categorical encoding** | One‑hot/ordinal encoding for *sex*, *chest pain type*, etc. | Enables tree‑based algorithms to interpret categorical information |
| **Feature scaling** | Standardized numeric columns (age, cholesterol, etc.) | Improves convergence for algorithms that are distance‑sensitive (if later used) |
| **Class balancing** | Applied SMOTE (oversampling) / class‑weighting | Mitigates the impact of any imbalance in the target variable |
| **Train/validation/test split** | 70 % / 15 % / 15 % stratified split | Ensures unbiased performance estimates |
| **Feature engineering** | Added BMI (weight/height²) and age bins | Provides potentially more predictive signals |

The cleaned dataset grew from 5 to 6 columns after adding engineered features.

---

**Model Performance**  

| Algorithm | Accuracy | F1‑Score | Comments |
|-----------|----------|----------|----------|
| Random Forest (default hyper‑parameters) | **0.425** | **0.420** | Performance is markedly low; likely due to limited sample size, insufficient feature richness, or sub‑optimal hyper‑parameters. |

---

**Recommendations**  

1. **Expand the Dataset**  
   * Acquire a larger, more representative heart‑disease cohort (e.g., full UCI Cleveland dataset ~303 records, or combine multiple sources).  
   * Include additional clinically relevant variables (blood pressure, cholesterol, smoking status, ECG results, etc.).

2. **Feature Enrichment & Selection**  
   * Perform exploratory analysis to identify high‑impact predictors.  
   * Consider domain‑driven features such as **Framingham risk score components** or interaction terms.

3. **Model Tuning & Alternatives**  
   * Conduct systematic hyper‑parameter optimization (grid/random search, Bayesian optimization).  
   * Test algorithms better suited for small tabular data: Gradient Boosting (XGBoost, LightGBM), Logistic Regression with regularization, or Support Vector Machines.

4. **Cross‑Validation & Robust Evaluation**  
   * Use stratified k‑fold CV (k = 5–10) to obtain stable performance estimates.  
   * Report additional metrics (ROC‑AUC, sensitivity, specificity) critical for medical decision‑making.

5. **Address Class Imbalance Rigorously**  
   * Verify the degree of imbalance; if severe, combine oversampling (SMOTE) with undersampling or use cost‑sensitive learning.

6. **Pipeline Automation**  
   * Encapsulate preprocessing, feature engineering, and modeling in a reproducible pipeline (e.g., scikit‑learn `Pipeline` or MLflow) to streamline experimentation.

7. **Clinical Validation**  
   * Before deployment, validate the model on an external hold‑out cohort and assess calibration (e.g., reliability diagrams).

Implementing these steps should markedly improve predictive performance and move the solution toward a clinically useful risk‑assessment tool.
```
## Agentic Performance Analysis & Technical Verification
### 1. Deterministic Multi-Agent Orchestration
The primary objective f evaluating this run was to verify whether the autonomous state graph executes continuously without pipeline breaks or manual intervention.
- **State Propogation:** Te state graph correctly transitioned across all 5 nodes (Planner -> Data Finder -> Data Prep -> Modeling -> Reporter).
- **Format Adherence:** Downstream code successfully parsed structural fields from the Groq API(openai/gpt-oss-120b) without JSON formatting errors or regex fallback failures.
- **Terminal Environment Hygiene:** The second execution confirmed clean runtime execution following deprecation adjustments to
  ```python
  select_dtypes(include=['str', 'object', 'category']).
  ```
### 2. Modeling Performance Context(Accuracy = 42.5%)
While an accuracy of 42.5% and an $F_1\text{-score}$ of 0.420 would be insufficient for production or clinical environments, this result validates that the system is operating without artificial data leakage:
- **Baseline Integrity:** The Data Finder Agent generated a small 200-row synthetic sample set(sample_data.csv) using uncorrelated random numbers.
- **No Overfitting/Leakage:** A model trained on pure noise yielding near-chance accuaracy proves that the Data Prep Agent and Modeling Agent isolated the train/test splits cleanly without data leakage (e.g., fitting scalers on test data prior to splitting).
- **Automated Diagnosis:** The Reporter Agent correctly recognized the low metrics, contextualized the failure as a data density problem rather than a pipeline error, and drafted appropriate mitigation strategies ((e.g., acquiring the full UCI Cleveland dataset, adding composite clinical features).

## Evaluation & Next Steps
### Key Achievements
- **Zero Deprecation Warnings:** Updated pandas feature selection ensure clean execution logs.
- **Deterministic Execution:** Safe regex-based JSON extraction guarantees stable state transitions between LLM steps and local execution code.
- **Fully Modular Pipeline:** Nodes operate independently, making it straightforward to plug in real external APIs or advanced hyperparameter tuning algorithms.


  
   
