from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_goal: str
    plan : [str, Any]
    dataset_path: str
    dataset_info: Dict[str, Any]
    cleaned_data_path: str
    eda_summary: Dict[str, Any]
    model_results: Dict[str, Any]
    final_report: str
    

