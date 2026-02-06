"""Simple Pinecone client setup and index creation helpers."""

from __future__ import annotations

import os
from typing import Any

from pinecone import Pinecone, ServerlessSpec


def get_pinecone_index(
    index_name: str,
    *,
    vector_dim: int = 3072,
    similarity_option: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> Any:
    """Create the Pinecone index if missing and return the index handle."""

    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is not set.")

    pc = Pinecone(
        api_key=pinecone_key
    )

    if index_name not in pc.list_indexes().names():
        print(f"Creating index {index_name}")
        pc.create_index(
            name=index_name,
            dimension=vector_dim,
            metric=similarity_option,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    return pc.Index(name=index_name)
