"""Graph node implementations for the main chat graph."""

from __future__ import annotations

import json
from typing import Any

from src.core.gemini_client import get_default_gemini_manager
from src.graph.helpers import (
    format_chat_history,
    format_retrieval_context,
    parse_intent,
)
from src.graph.state import ChatState
from src.prompts.chain import select_next_prompt
from src.prompts.templates import (
    FORM_CONFIRM_PROMPT,
    FORM_DISCOVERY_PROMPT,
    FORM_FILL_PROMPT,
    FORM_INTENT_PROMPT,
    FORM_QA_PROMPT,
)
from src.rag.retriever import build_retriever_graph


def detect_intent(state: ChatState) -> ChatState:
    """Run Prompt 1 to classify user intent."""

    manager = get_default_gemini_manager()
    chat_history = format_chat_history(state.get("chat_history"))
    prompt = FORM_INTENT_PROMPT.format(
        message=state["new_message"],
        chat_history=chat_history,
    )
    raw = manager.generate_text(prompt)

    return {**state, "intent": parse_intent(raw)}


def run_retriever(state: ChatState) -> ChatState:
    """Invoke the retriever sub-graph to gather context."""

    retriever_graph = build_retriever_graph()
    out = retriever_graph.invoke(
        {
            "query": state["new_message"],
            "form": state.get("form"),
            "chat_history": state.get("chat_history"),
            "intent": state.get("intent"),
        }
    )

    return {**state, "retrieval_results": out.get("retrieval_results", {})}


def generate_response(state: ChatState) -> ChatState:
    """Select a prompt template and generate the final response."""

    def format_fields(value: Any) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        except TypeError:
            return str(value)

    manager = get_default_gemini_manager()
    intent = state.get("intent") or "other"
    form_known = bool(state.get("form"))

    prompt_name = select_next_prompt(intent, form_known)
    context = format_retrieval_context(state.get("retrieval_results", {}))

    if prompt_name == "FORM_DISCOVERY_PROMPT":
        chat_history = format_chat_history(state.get("chat_history"))
        prompt = FORM_DISCOVERY_PROMPT.format(
            message=state["new_message"],
            current_hint=state.get("form") or "unknown",
            chat_history=chat_history,
            context=context or "No relevant context found.",
        )
    elif prompt_name == "FORM_QA_PROMPT":
        chat_history = format_chat_history(state.get("chat_history"))
        prompt = FORM_QA_PROMPT.format(
            message=state["new_message"],
            form_title=state.get("form") or "unknown",
            context=context or "No relevant context found.",
            chat_history=chat_history,
        )
    elif prompt_name == "FORM_FILL_PROMPT":
        chat_history = format_chat_history(state.get("chat_history"))
        prompt = FORM_FILL_PROMPT.format(
            form_title=state.get("form") or "unknown",
            known_fields=format_fields(state.get("known_fields")),
            remaining_fields=format_fields(state.get("remaining_fields")),
            chat_history=chat_history,
            context=context or "No relevant context found.",
        )
    else:
        prompt = FORM_CONFIRM_PROMPT.format(
            form_title=state.get("form") or "unknown",
            filled_fields=format_fields(state.get("filled_fields")),
        )

    response_text = manager.generate_text(prompt)
    return {**state, "response_text": response_text}
