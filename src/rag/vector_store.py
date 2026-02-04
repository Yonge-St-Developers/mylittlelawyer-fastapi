"""Vector store abstraction (Pinecone implementation)."""

from __future__ import annotations

from typing import Any, Iterable

from src.core.pinecone_client import get_pinecone_index


def upsert_vectors(
    vectors: Iterable[dict[str, Any]],
    *,
    index_name: str,
    vector_dim: int = 3072,
    similarity_option: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> Any:
    """Upsert vectors into Pinecone."""

    index = get_pinecone_index(
        index_name,
        vector_dim=vector_dim,
        similarity_option=similarity_option,
        cloud=cloud,
        region=region,
    )
    return index.upsert(vectors=list(vectors))


def query_vectors(
    vector: list[float],
    *,
    index_name: str,
    top_k: int = 3,
    vector_dim: int = 3072,
    similarity_option: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> Any:
    """Query Pinecone for similar vectors."""

    index = get_pinecone_index(
        index_name,
        vector_dim=vector_dim,
        similarity_option=similarity_option,
        cloud=cloud,
        region=region,
    )
    return index.query(vector=vector, top_k=top_k, include_metadata=True)
