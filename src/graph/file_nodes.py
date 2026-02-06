"""Graph node implementations for file generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from src.core.gemini_client import get_default_gemini_manager
from src.graph.file_state import FileState
from src.prompts.templates import FORM_FILE_JSON_PROMPT


def generate_file_json(state: FileState) -> FileState:
    """Generate a JSON payload for PDF generation."""

    chat_history = _format_chat_history(state.get("chat_history") or [])
    prompt = FORM_FILE_JSON_PROMPT.format(
        form_title=state.get("form_title") or "unknown",
        chat_history=chat_history or "No chat history provided.",
    )

    manager = get_default_gemini_manager()
    raw = manager.generate_text(prompt)
    file_json = _safe_json_loads(raw)

    return {**state, "file_json": file_json}


def _format_chat_history(history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for item in history:
        role = item.get("role", "unknown")
        content = item.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.strip()}
