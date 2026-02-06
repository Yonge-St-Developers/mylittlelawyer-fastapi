"""Typed state for the file generation graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class FileState(TypedDict, total=False):
    """State passed through the file generation graph."""

    session_id: str
    form_title: str
    chat_history: Optional[List[Dict[str, str]]]

    file_json: Dict[str, Any]
