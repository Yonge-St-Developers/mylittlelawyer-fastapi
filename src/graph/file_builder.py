"""Builder for the file generation graph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.file_nodes import generate_file_json
from src.graph.file_state import FileState


def build_file_graph():
    """Create the file generation graph for /ai/file endpoint."""

    graph = StateGraph(FileState)
    graph.add_node("generate_file_json", generate_file_json)

    graph.add_edge("generate_file_json", END)
    graph.set_entry_point("generate_file_json")

    return graph.compile()
