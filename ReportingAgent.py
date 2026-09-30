def reporter_node(state: AgentState) -> Dict[str, Any]:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    prompt = f"""
    Write a concise executive summary report based on these pipeline findings:
    - User Goal: {state['user_goal']}
    - Execution Plan: {state['plan']}
    - Dataset Summary: Raw shape {state['dataset_info']['raw_shape']}, Cleaned shape {state['eda_summary']['cleaned_shape']}
    - Model Performance: {state['model_results']}
    
    Structure with sections: Executive Summary, Data Preprocessing Actions, Model Performance, Recommendations.
    """
    
    response = llm.invoke(prompt)
    return {"final_report": response.content}