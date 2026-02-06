"""Backwards-compatible entrypoint for the chat graph."""

from __future__ import annotations

from src.graph.builder import build_chat_graph
from src.graph.state import ChatState

__all__ = ["ChatState", "build_chat_graph"]
