import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def fetchDatasetFromKaggle(query: str, download_dir: str = "./data") -> str:
    api = KaggleApi()
    api.authenticate()
    datasets = api.dataset_list(search = query)

    if not datasets:
        raise ValueError(f"No datasets found in: {query}")

    top_dataset = datasets[0].ref
    api.dataset_download_files(top_dataset, path=download_dir, unzip=True)
    for file in os.listdir(download_dir):
        if file.endswith('.csv'):
            return os.path.join(download_dir, file)
    raise FileNotFoundError("No CSV found in downloaded package.")


def data_finder_node(state: AgentState) -> Dict[str, Any]:
    keywords = " ".join(state['plan']['search_keywords'])
    csv_path = fetch_dataset_from_kaggle(query=keywords)
    df = pd.read_csv(csv_path)
    info = {
        "raw_shape": list(df.shape),
        "columns": list(df.columns),
        "missing_counts": df.isnull().sum().to_dict()
    }
    return {"dataset_path": csv_path, "dataset_info": info}