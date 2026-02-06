"""Helper utilities for the main chat graph."""

from __future__ import annotations

from typing import Any, Dict, List, Literal


IntentType = Literal[
    "answering_form_field",
    "asking_about_form",
    "describing_situation",
    "other",
]


def parse_intent(raw: str) -> IntentType:
    """
    Very simple parser for intent classification output.

    NOTE: Replace with robust JSON parsing when you finalize the prompt format.
    """

    raw_lower = raw.lower()
    if "answering_form_field" in raw_lower:
        return "answering_form_field"
    if "asking_about_form" in raw_lower:
        return "asking_about_form"
    if "describing_situation" in raw_lower:
        return "describing_situation"
    return "other"


def format_retrieval_context(results: Dict[str, Any]) -> str:
    """
    Normalize Pinecone results into a single context string.

    This keeps model prompts consistent across indexes.
    """

    context_chunks: List[str] = []
    for index_name, result in (results or {}).items():
        matches = getattr(result, "matches", []) or []
        for match in matches:
            meta = getattr(match, "metadata", {}) or {}
            text = meta.get("text") or ""
            if text:
                context_chunks.append(f"[{index_name}] {text}")

    return "\n\n".join(context_chunks).strip()
