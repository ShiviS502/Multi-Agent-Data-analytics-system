from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

class ExecutionPlan(BaseModel):
    task_type: str = Field(description="'classification' or 'regression'")
    target_column: str = Field(description="Name of target variable to predict")
    search_keywords: List[str] = Field(description="Keywords to locate dataset")
    prep_notes: str = Field(description="Special handling required for data prep")

def planner_node (state: AgentState) -> Dict[str, Any]:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(ExecutionPlan)
    
    prompt = f"Analyze this goal and build an execution plan: {state['user_goal']}"
    plan: ExecutionPlan = structured_llm.invoke(prompt)
    
    return {"plan": plan.model_dump()}

