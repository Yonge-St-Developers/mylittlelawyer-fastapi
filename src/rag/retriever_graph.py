"""Utilities for visualizing the retriever LangGraph."""

from __future__ import annotations

from networkx import display
from IPython.display import Image, display
from src.rag.retriever import build_retriever_graph


def get_retriever_graph_figutre() -> str:
    """Return a Mermaid diagram of the retriever graph."""

    compiled = build_retriever_graph()

    mermid_graph = compiled.get_graph().draw_mermaid()

    display(Image(compiled.get_graph().draw_mermaid_png()))

    return mermid_graph