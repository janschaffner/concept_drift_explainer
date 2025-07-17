import sys
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import Literal
import os
from functools import partial
from pinecone import Pinecone

# --- Path Correction ---
# Ensures that the script can correctly import modules from the 'backend' directory
# by adding the project's root directory to the system's path.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# -----------------------

from dotenv import load_dotenv

# Import the shared state schema and all agent functions
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
    Determines the next node to call after the main explanation is generated.
    
    This function acts as a router for the graph's conditional logic. If the user
    has asked a follow-up question (which is present in the state), it directs
    the workflow to the `chatbot_agent`. Otherwise, it signals the end of the
    execution.

    Args:
        state: The current state of the graph.

    Returns:
        A string literal indicating the next node to execute ("chatbot_agent" or END).
    """
    if state.get("user_question"):
        return "chatbot_agent"
    else:
        return END

def build_graph():
    """
    Builds and compiles the complete, conditional LangGraph for the concept drift
    explanation system.

    This function defines the architecture of the agentic workflow, registering each
    agent as a node and defining the sequence of execution through directed edges.

    Returns:
        A compiled LangGraph application that is ready to be executed.
    """
    # --- Centralized Pinecone Initialization ---
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    if not all([api_key, pinecone_index_name]):
        raise ValueError("Pinecone API key or index name not found in .env file.")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(pinecone_index_name)
    # -----------------------------------------

    # Bind the index object to the agents that need it
    retrieval_agent_with_index = partial(run_context_retrieval_agent, index=index)
    explanation_agent_with_index = partial(run_explanation_agent, index=index)

    # Initialize a new state graph with the custom GraphState schema.
    workflow = StateGraph(GraphState)

    # Add each agent function as a node in the graph.
    # Each node is a step in the pipeline that can read from and write to the shared state.
    workflow.add_node("drift_agent", run_drift_agent)
    workflow.add_node("context_retrieval_agent", retrieval_agent_with_index)
    workflow.add_node("re_ranker_agent", run_reranker_agent)
    workflow.add_node("franzoi_mapper_agent", run_franzoi_mapper_agent)
    workflow.add_node("explanation_agent", explanation_agent_with_index)
    workflow.add_node("chatbot_agent", run_chatbot_agent)


    # Define the sequence of execution
    # This sets up the directed edges that control the main analytical pipeline.
    workflow.set_entry_point("drift_agent")
    workflow.add_edge("drift_agent", "context_retrieval_agent")
    workflow.add_edge("context_retrieval_agent", "re_ranker_agent")
    workflow.add_edge("re_ranker_agent", "franzoi_mapper_agent")
    workflow.add_edge("franzoi_mapper_agent", "explanation_agent")
    
    # Define the conditional logic for the interactive chatbot loop
    # After the main explanation is generated, the `should_continue` router is called.
    workflow.add_conditional_edges(
        # The routing logic starts from the 'explanation_agent' node.
        "explanation_agent",
        # The `should_continue` function determines the path.
        should_continue,
        # This dictionary maps the function's output to the next node.
        {
            "chatbot_agent": "chatbot_agent",
            "__end__": END
        }
    )
    # After the chatbot answers a question, it loops back to the explanation agent,
    # allowing for a continuous conversation.
    workflow.add_edge("chatbot_agent", "explanation_agent")


    # Compile the graph into a runnable application object.
    app = workflow.compile()
    
    print("✅ LangGraph workflow compiled successfully!")
    
    return app

if __name__ == '__main__':
    # This block allows the script to be run directly for testing or visualization.
    app = build_graph()

    # To visualize the graph, you can uncomment the following lines.
    # This requires `mermaid-cli` and `graphviz` to be installed.
    # from IPython.display import Image, display
    # try:
    #     img_data = app.get_graph().draw_mermaid_png()
    #     display(Image(img_data))
    # except Exception as e:
    #     print(f"Could not draw graph: {e}")