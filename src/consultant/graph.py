"""LangGraph pipeline for the consultant endpoint (/ai/consultant)."""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.consultant.local_indexer import index_local_pdfs
from src.consultant.prompts import CONSULTANT_PROMPT
from src.consultant.retriever import retrieve_cases
from src.core.gemini_client import get_default_gemini_manager
from src.graph.helpers import format_chat_history


class ConsultantState(TypedDict, total=False):
    """State for consultant RAG graph."""

    message: str
    chat_history: Optional[List[Dict[str, str]]]
    refresh_index: bool
    retrieval: Dict
    response_text: str


def maybe_crawl(state: ConsultantState) -> ConsultantState:
    """Optionally refresh the case index from CanLII."""

    if state.get("refresh_index"):
        index_local_pdfs()
    return state


def retrieve(state: ConsultantState) -> ConsultantState:
    """Retrieve relevant case chunks from ChromaDB."""

    result = retrieve_cases(state["message"], top_k=10)
    return {**state, "retrieval": result}


def generate(state: ConsultantState) -> ConsultantState:
    """Generate the final response using Gemini."""

    manager = get_default_gemini_manager()
    chat_history = format_chat_history(state.get("chat_history"))

    # Flatten documents for prompt context
    docs = (state.get("retrieval") or {}).get("documents") or []
    flat_docs = "\n\n".join([item for sub in docs for item in sub]) if docs else ""

    prompt = CONSULTANT_PROMPT.format(
        message=state["message"],
        chat_history=chat_history,
        context=flat_docs or "No relevant context found.",
    )

    response_text = manager.generate_text(prompt)
    return {**state, "response_text": response_text}


def build_consultant_graph():
    """Build the consultant RAG graph."""

    graph = StateGraph(ConsultantState)
    graph.add_node("maybe_crawl", maybe_crawl)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge("maybe_crawl", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    graph.set_entry_point("maybe_crawl")

    return graph.compile()
