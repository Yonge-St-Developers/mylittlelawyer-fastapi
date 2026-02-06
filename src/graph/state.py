"""Typed state for the main chat graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from src.graph.helpers import IntentType


class ChatState(TypedDict, total=False):
    """State passed through the chat graph."""

    session_id: str
    new_message: str
    chat_history: Optional[List[Dict[str, str]]]
    form: Optional[str]

    intent: Optional[IntentType]
    retrieval_results: Dict[str, Any]
    known_fields: Any
    remaining_fields: Any
    filled_fields: Any

    response_text: str
