"""LangGraph pipeline for the consultant endpoint (/ai/consultant)."""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.consultant.local_indexer import index_local_pdfs
from src.consultant.prompts import CONSULTANT_PROMPT
from src.consultant.retriever import retrieve_cases, retrieve_pinecone_context
from src.core.gemini_client import get_default_gemini_manager
from src.graph.helpers import format_chat_history


class ConsultantState(TypedDict, total=False):
    """State for consultant RAG graph."""

    message: str
    chat_history: Optional[List[Dict[str, str]]]
    refresh_index: bool
    retrieval: Dict
    pinecone_retrieval: Dict
    response_text: str


def maybe_crawl(state: ConsultantState) -> ConsultantState:
    """Optionally refresh the case index from CanLII."""

    if state.get("refresh_index"):
        index_local_pdfs()
    return state


def retrieve(state: ConsultantState) -> ConsultantState:
    """Retrieve relevant case chunks from ChromaDB."""

    chroma_result = retrieve_cases(state["message"], top_k=10)
    pinecone_result = retrieve_pinecone_context(state["message"], top_k=10)
    return {**state, "retrieval": chroma_result, "pinecone_retrieval": pinecone_result}


def generate(state: ConsultantState) -> ConsultantState:
    """Generate the final response using Gemini."""

    manager = get_default_gemini_manager()
    chat_history = format_chat_history(state.get("chat_history"))

    # Flatten documents for prompt context
    docs = (state.get("retrieval") or {}).get("documents") or []
    flat_docs = "\n\n".join([item for sub in docs for item in sub]) if docs else ""

    # Add Pinecone context (laws/instructions)
    pinecone_ctx: list[str] = []
    for index_name, result in (state.get("pinecone_retrieval") or {}).items():
        matches = getattr(result, "matches", []) or []
        for match in matches:
            meta = getattr(match, "metadata", {}) or {}
            text = meta.get("text") or ""
            if text:
                pinecone_ctx.append(f"[{index_name}] {text}")

    pinecone_context = "\n\n".join(pinecone_ctx).strip()
    combined_context = "\n\n".join(
        [c for c in [pinecone_context, flat_docs] if c]
    ) or "No relevant context found."

    prompt = CONSULTANT_PROMPT.format(
        message=state["message"],
        chat_history=chat_history,
        context=combined_context,
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
