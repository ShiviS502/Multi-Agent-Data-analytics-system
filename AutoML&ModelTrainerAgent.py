from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,f1_score, mean_squared_error, r2_score

def modeling_node(state:AgentState) -> Dict[str,Any]:
    df = pd.read_csv(state['cleaned_data_path'])
    target_col = state['plan']['target_column']
    task_type = state['plan']['task_type']

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_score=0.2, random_state=42)
    metrics = {}
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators = 100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {"accuracy": round(float(accuracy_score(y_test, preds)), 4), 
                   "f1_score": round(float(f1_score(y_test, preds, average = "weighted")), 4)
                   }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state =42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {"rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4), 
                   "r2_score": round(float(r2_score(y_test, preds)), 4)
                   }

    return {"model_results": {"algorithm": "RandomForest" , "metrics": metrics}}