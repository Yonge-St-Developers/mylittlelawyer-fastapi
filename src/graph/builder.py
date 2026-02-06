"""Builder for the main chat graph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.nodes import detect_intent, generate_response, run_retriever
from src.graph.state import ChatState


def build_chat_graph():
    """Create the full chat graph for /ai/chat endpoint."""

    graph = StateGraph(ChatState)
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("run_retriever", run_retriever)
    graph.add_node("generate_response", generate_response)

    graph.add_edge("detect_intent", "run_retriever")
    graph.add_edge("run_retriever", "generate_response")
    graph.add_edge("generate_response", END)

    graph.set_entry_point("detect_intent")

    return graph.compile()
