"""Prompt template for consultant RAG."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

CONSULTANT_PROMPT = PromptTemplate(
    input_variables=["message", "context", "chat_history"],
    template=(
        "You are an Ontario LTB legal consultant. "
        "Use the retrieved case context to answer the user's question. "
        "Be precise, cite relevant facts, and keep the answer practical. "
        "If the context is insufficient, say so and ask a clarifying question.\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Retrieved context:\n{context}\n\n"
        "User message: {message}\n\n"
        "Answer:" 
    ),
)
