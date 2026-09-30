from langgraph.graph import StateGraph, END, START

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
        "user_goal": "Find a dataset on heart disease, clean it, and build a classifier to predict diagnosis."
    }
    
    final_output = app.invoke(initial_state)
    print(final_output["final_report"])