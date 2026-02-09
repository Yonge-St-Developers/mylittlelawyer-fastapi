"""Retriever for consultant RAG using ChromaDB + Pinecone."""

from __future__ import annotations

from typing import Any

import os

from src.consultant.chroma_client import chroma_query, get_or_create_collection
from src.core.pinecone_client import get_pinecone_index
from src.rag.embedder import embed_query


def retrieve_cases(query: str, top_k: int = 10, collection_name: str | None = None) -> Any:
    """Retrieve most relevant case chunks from ChromaDB."""

    collection_id, _ = get_or_create_collection(collection_name)
    vector = embed_query(query)
    return chroma_query(collection_id, query_embeddings=[vector], n_results=top_k)


def _pinecone_indexes_for_consultant() -> dict[str, str]:
    return {
        "rta_act": os.getenv("INDEX_RTA_ACT", "").strip(),
        "sppa_act": os.getenv("INDEX_SPPA_ACT", "").strip(),
        "ltb_rules": os.getenv("INDEX_LTB_RULES", "").strip(),
        "ltb_practice_directions": os.getenv("INDEX_LTB_PRACTICE_DIRECTIONS", "").strip(),
        "ltb_guidelines": os.getenv("INDEX_LTB_GUIDELINES", "").strip(),
    }


def retrieve_pinecone_context(query: str, top_k: int = 10) -> dict:
    """Retrieve from the 5 Pinecone law/instruction indexes."""

    vector = embed_query(query)
    results: dict = {}
    for key, idx in _pinecone_indexes_for_consultant().items():
        if not idx:
            continue
        index = get_pinecone_index(idx, vector_dim=3072)
        results[key] = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results
