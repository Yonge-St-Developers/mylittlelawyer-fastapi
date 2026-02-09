"""Retriever for consultant RAG using ChromaDB."""

from __future__ import annotations

from typing import Any

from src.consultant.chroma_client import chroma_query, get_or_create_collection
from src.rag.embedder import embed_query


def retrieve_cases(query: str, top_k: int = 10, collection_name: str | None = None) -> Any:
    """Retrieve most relevant case chunks from ChromaDB."""

    collection_id, _ = get_or_create_collection(collection_name)
    vector = embed_query(query)
    return chroma_query(collection_id, query_embeddings=[vector], n_results=top_k)
