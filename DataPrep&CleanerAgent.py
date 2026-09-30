import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def data_prep_node(state: AgentState) -> Dict[str, Any]:
    df = pd.read_csv(state['dataset_path'])
    target_col = state['plan']['target_column']
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
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
    return {"cleaned_data_path": cleaned_path, "eda_summary": eda_summary}