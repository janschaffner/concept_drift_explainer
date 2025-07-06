import sys
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import Literal

# --- Path Correction ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from backend.state.schema import GraphState
from backend.agents.drift_agent import run_drift_agent
from backend.agents.context_retrieval_agent import run_context_retrieval_agent
from backend.agents.re_ranker_agent import run_reranker_agent
from backend.agents.franzoi_mapper_agent import run_franzoi_mapper_agent
from backend.agents.explanation_agent import run_explanation_agent
from backend.agents.chatbot_agent import run_chatbot_agent

# --- Router function for the chatbot loop ---
def should_continue(state: GraphState) -> Literal["chatbot_agent", "__end__"]:
    """
    This is our router. It determines the next node to call based on the state.
    If the user has asked a question, it routes to the chatbot.
    Otherwise, it ends the conversation.
    """
    if state.get("user_question"):
        return "chatbot_agent"
    else:
        return END

def build_graph():
    """
    Builds the LangGraph agent-based workflow.
    """
    workflow = StateGraph(GraphState)

    # Add nodes for each agent
    # Each node corresponds to a function that modifies the state
    workflow.add_node("drift_agent", run_drift_agent)
    workflow.add_node("context_retrieval_agent", run_context_retrieval_agent)
    workflow.add_node("re_ranker_agent", run_reranker_agent)
    workflow.add_node("franzoi_mapper_agent", run_franzoi_mapper_agent)
    workflow.add_node("explanation_agent", run_explanation_agent)
    workflow.add_node("chatbot_agent", run_chatbot_agent)


    # --- Define the sequence of execution ---
    # This sets up the directed edges of the graph
    workflow.set_entry_point("drift_agent")
    workflow.add_edge("drift_agent", "context_retrieval_agent")
    # The new Re-Ranker agent runs after retrieval
    workflow.add_edge("context_retrieval_agent", "re_ranker_agent")
    # The Franzoi Mapper now runs on the re-ranked context
    workflow.add_edge("re_ranker_agent", "franzoi_mapper_agent")
    workflow.add_edge("franzoi_mapper_agent", "explanation_agent")
    
    # --- CONDITIONAL LOGIC for the chatbot ---
    # After the explanation, we check if the user has asked a question
    workflow.add_conditional_edges(
        "explanation_agent",
        should_continue,
        {
            "chatbot_agent": "chatbot_agent",
            "__end__": END
        }
    )
    # After the chatbot runs, it loops back, allowing for more questions
    workflow.add_edge("chatbot_agent", "explanation_agent")


    # Compile the graph into a runnable object
    app = workflow.compile()
    
    print("✅ LangGraph workflow compiled successfully!")
    
    return app

if __name__ == '__main__':
    # This allows us to run this file directly to build and test the graph
    app = build_graph()

    # To visualize the graph, you can uncomment the following lines
    # Make sure you have mermaid-cli installed (`npm install -g @mermaid-js/mermaid-cli`)
    # and graphviz (`conda install python-graphviz`)
    
    # from IPython.display import Image, display
    # try:
    #     img_data = app.get_graph().draw_mermaid_png()
    #     display(Image(img_data))
    # except Exception as e:
    #     print(f"Could not draw graph: {e}")