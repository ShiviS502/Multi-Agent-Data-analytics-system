import os
import json
import re
from typing import TypedDict, Any, Dict, List
import pandas as pd
import numpy as np

from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START

GROQ_MODEL = "openai/gpt-oss-120b"


# ==========================================
# 1. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    user_goal: str
    plan: Dict[str, Any]
    dataset_path: str
    dataset_info: Dict[str, Any]
    cleaned_data_path: str
    eda_summary: Dict[str, Any]
    model_results: Dict[str, Any]
    final_report: str


# ==========================================
# 2. PLANNER AGENT (Robust JSON Parsing)
# ==========================================
def planner_node(state: AgentState) -> Dict[str, Any]:
    print("\n[🤖 Planner Agent] Analyzing goal and structuring execution plan using Groq...")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    
    prompt = f"""You are an expert data science project planner.
Analyze this goal: "{state['user_goal']}"

Return ONLY a valid JSON object (no markdown, no extra text) matching this format:
{{
    "task_type": "classification or regression",
    "target_column": "name of target column to predict",
    "search_keywords": ["keyword1", "keyword2"],
    "prep_notes": "notes on data preparation"
}}
"""
    response = llm.invoke(prompt)
    
    # Clean output to ensure pure JSON parsing
    raw_text = response.content.strip()
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        plan_dict = json.loads(json_match.group(0))
    else:
        plan_dict = {
            "task_type": "classification",
            "target_column": "target",
            "search_keywords": ["data"],
            "prep_notes": "Default fallback plan"
        }
    
    print(f"[🤖 Planner Agent] Target Column: '{plan_dict.get('target_column')}' | Task Type: '{plan_dict.get('task_type')}'")
    return {"plan": plan_dict}


# ==========================================
# 3. DATA FINDER AGENT
# ==========================================
def data_finder_node(state: AgentState) -> Dict[str, Any]:
    print("\n[🔍 Data Finder Agent] Fetching/generating dataset...")
    os.makedirs("./data", exist_ok=True)
    csv_path = "./data/sample_data.csv"
    
    target_col = state['plan'].get('target_column', 'target')
    task_type = state['plan'].get('task_type', 'classification')
    
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'age': np.random.randint(25, 75, size=n_samples),
        'cholesterol': np.random.normal(210, 30, size=n_samples),
        'blood_pressure': np.random.normal(120, 15, size=n_samples),
        'gender': np.random.choice(['Male', 'Female'], size=n_samples),
    }
    
    # Introduce missing values to test data prep
    data['cholesterol'][10:20] = np.nan
    
    if task_type.lower() == 'classification':
        data[target_col] = np.random.choice([0, 1], size=n_samples)
    else:
        data[target_col] = data['age'] * 1.5 + np.random.normal(0, 5, size=n_samples)
        
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    info = {
        "raw_shape": list(df.shape),
        "columns": list(df.columns),
        "missing_counts": df.isnull().sum().to_dict()
    }
    print(f"[🔍 Data Finder Agent] Dataset prepared at '{csv_path}' with shape {info['raw_shape']}")
    return {"dataset_path": csv_path, "dataset_info": info}


# ==========================================
# 4. DATA PREP & CLEANER AGENT
# ==========================================
def data_prep_node(state: AgentState) -> Dict[str, Any]:
    print("\n[🧹 Data Prep Agent] Cleaning, imputing, and scaling dataset...")
    df = pd.read_csv(state['dataset_path'])
    target_col = state['plan'].get('target_column', 'target')
    
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['str', 'object', 'category']).columns.tolist()
    
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        X[num_cols] = num_imputer.fit_transform(X[num_cols])
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), encoded_df], axis=1)

    cleaned_path = "./data/cleaned_dataset.csv"
    cleaned_df = pd.concat([X, y.reset_index(drop=True)], axis=1)
    cleaned_df.to_csv(cleaned_path, index=False)
    
    eda_summary = {
        "cleaned_shape": list(cleaned_df.shape),
        "num_features_processed": len(num_cols),
        "cat_features_processed": len(cat_cols)
    }
    print(f"[🧹 Data Prep Agent] Cleaned dataset saved to '{cleaned_path}'. Processed {len(num_cols)} numeric & {len(cat_cols)} categorical features.")
    return {"cleaned_data_path": cleaned_path, "eda_summary": eda_summary}


# ==========================================
# 5. AUTOML & MODELING AGENT
# ==========================================
def modeling_node(state: AgentState) -> Dict[str, Any]:
    print("\n[📊 Modeling Agent] Training model and calculating metrics...")
    df = pd.read_csv(state['cleaned_data_path'])
    target_col = state['plan'].get('target_column', 'target')
    
    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    task_type = state['plan'].get('task_type', 'classification')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task_type.lower() == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "f1_score": round(float(f1_score(y_test, preds, average='weighted')), 4)
        }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "r2_score": round(float(r2_score(y_test, preds)), 4)
        }
        
    print(f"[📊 Modeling Agent] Model Trained: RandomForest | Metrics: {metrics}")
    return {"model_results": {"algorithm": "RandomForest", "metrics": metrics}}


# ==========================================
# 6. REPORTING AGENT
# ==========================================
def reporter_node(state: AgentState) -> Dict[str, Any]:
    print("\n[📝 Reporter Agent] Compiling final summary report using Groq...")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    
    prompt = f"""
Write a concise executive summary report based on these pipeline findings:
- User Goal: {state['user_goal']}
- Execution Plan: {state['plan']}
- Dataset Summary: Raw shape {state['dataset_info']['raw_shape']}, Cleaned shape {state['eda_summary']['cleaned_shape']}
- Model Performance: {state['model_results']}

Structure with clear headers:
1. Executive Summary
2. Data Preprocessing Actions
3. Model Performance
4. Recommendations
"""
    
    response = llm.invoke(prompt)
    return {"final_report": response.content}


# ==========================================
# 7. GRAPH ORCHESTRATION & EXECUTION
# ==========================================
def build_pipeline():
    builder = StateGraph(AgentState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("data_finder", data_finder_node)
    builder.add_node("data_prep", data_prep_node)
    builder.add_node("modeling", modeling_node)
    builder.add_node("reporter", reporter_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "data_finder")
    builder.add_edge("data_finder", "data_prep")
    builder.add_edge("data_prep", "modeling")
    builder.add_edge("modeling", "reporter")
    builder.add_edge("reporter", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_pipeline()
    
    initial_state = {
        "user_goal": "Find heart disease data, preprocess it, and build a model predicting heart disease risk."
    }
    
    final_output = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("FINAL EXECUTIVE REPORT")
    print("="*50)
    print(final_output["final_report"])