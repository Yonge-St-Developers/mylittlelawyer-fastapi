"""Document retrieval logic for RAG workflows."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.core.pinecone_client import get_pinecone_index
from src.rag.embedder import embed_query


IntentType = Literal[
    "answering_form_field",
    "asking_about_form",
    "describing_situation",
    "other",
]


class RetrieverState(TypedDict, total=False):
    """State passed through the retriever graph."""

    query: str
    form: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    intent: Optional[IntentType]
    retrieval_results: Dict[str, Any]


def _get_index_envs() -> Dict[str, str]:
    """Load index names from environment variables."""

    return {
        "rta_act": os.getenv("INDEX_RTA_ACT", "").strip(),
        "sppa_act": os.getenv("INDEX_SPPA_ACT", "").strip(),
        "ltb_rules": os.getenv("INDEX_LTB_RULES", "").strip(),
        "ltb_practice_directions": os.getenv("INDEX_LTB_PRACTICE_DIRECTIONS", "").strip(),
        "ltb_guidelines": os.getenv("INDEX_LTB_GUIDELINES", "").strip(),
        "ltb_forms": os.getenv("INDEX_LTB_FORMS", "").strip(),
        "form_instructions": os.getenv("INDEX_FORM_INSTRUCTIONS", "").strip(),
    }


def _require_index(name: str) -> str:
    if not name:
        raise ValueError("Missing index name in environment variables.")
    return name


def _query_index(index_name: str, vector: list[float], top_k: int = 10) -> Any:
    """Query a Pinecone index and return matches."""

    index = get_pinecone_index(index_name, vector_dim=3072)
    return index.query(vector=vector, top_k=top_k, include_metadata=True)


def retrieve_all_indexes(state: RetrieverState) -> RetrieverState:
    """
    Retrieve from all 7 indexes.

    Used for the first message when form is unknown.
    """

    indices = _get_index_envs()
    query_vector = embed_query(state["query"])

    results: Dict[str, Any] = {}
    for key, idx in indices.items():
        if not idx:
            continue
        results[key] = _query_index(_require_index(idx), query_vector, top_k=10)

    return {**state, "retrieval_results": results}


def retrieve_form_instructions_only(state: RetrieverState) -> RetrieverState:
    """
    Retrieve only from the form instructions index.

    Used when a form is already identified and we only need procedure guidance.
    """

    indices = _get_index_envs()
    form_instructions = _require_index(indices["form_instructions"])
    query_vector = embed_query(state["query"])

    results = {
        "form_instructions": _query_index(form_instructions, query_vector, top_k=10)
    }
    return {**state, "retrieval_results": results}


def retrieve_six_indexes_for_form_questions(state: RetrieverState) -> RetrieverState:
    """
    Retrieve from the 6 knowledge indexes (excluding form_instructions).

    Used when the model detects the user is asking about form details.
    """

    indices = _get_index_envs()
    query_vector = embed_query(state["query"])

    results: Dict[str, Any] = {}
    for key in [
        "rta_act",
        "sppa_act",
        "ltb_rules",
        "ltb_practice_directions",
        "ltb_guidelines",
        "ltb_forms",
    ]:
        idx = indices.get(key, "")
        if not idx:
            continue
        results[key] = _query_index(_require_index(idx), query_vector, top_k=10)

    return {**state, "retrieval_results": results}


def route_retriever(state: RetrieverState) -> str:
    """
    Conditional router for retrieval strategy.

    Rules:
    - If form is None: retrieve all 7 indexes (first message / form discovery).
    - If intent == asking_about_form: retrieve 6 knowledge indexes.
    - If form is known: retrieve only form_instructions.
    """

    form_known = bool(state.get("form"))
    intent = state.get("intent")

    if not form_known:
        return "retrieve_all_indexes"

    if intent == "asking_about_form":
        return "retrieve_six_indexes_for_form_questions"

    return "retrieve_form_instructions_only"


def build_retriever_graph():
    """Build the conditional LangGraph retriever."""

    graph = StateGraph(RetrieverState)
    graph.add_node("retrieve_all_indexes", retrieve_all_indexes)
    graph.add_node("retrieve_six_indexes_for_form_questions", retrieve_six_indexes_for_form_questions)
    graph.add_node("retrieve_form_instructions_only", retrieve_form_instructions_only)

    graph.add_conditional_edges(
        "__start__",
        route_retriever,
        {
            "retrieve_all_indexes": "retrieve_all_indexes",
            "retrieve_six_indexes_for_form_questions": "retrieve_six_indexes_for_form_questions",
            "retrieve_form_instructions_only": "retrieve_form_instructions_only",
        },
    )

    graph.add_edge("retrieve_all_indexes", END)
    graph.add_edge("retrieve_six_indexes_for_form_questions", END)
    graph.add_edge("retrieve_form_instructions_only", END)

    return graph.compile()
