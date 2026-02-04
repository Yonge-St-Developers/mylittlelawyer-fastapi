"""Embedding creation pipeline (Gemini embeddings or compatible model)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

from src.core.gemini_client import GeminiClientManager, get_default_gemini_manager


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation."""

    batch_size: int = 32


def load_embedding_config() -> EmbeddingConfig:
    """Load embedding config from environment variables."""

    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))
    return EmbeddingConfig(batch_size=batch_size)


def _batch(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    """Yield lists of size N from an iterable."""

    batch_list: List[str] = []
    for item in iterable:
        batch_list.append(item)
        if len(batch_list) >= size:
            yield batch_list
            batch_list = []
    if batch_list:
        yield batch_list


def embed_texts(
    texts: list[str],
    config: EmbeddingConfig | None = None,
    manager: GeminiClientManager | None = None,
) -> list[list[float]]:
    """Embed a list of texts using the configured Gemini embedding model."""

    cfg = config or load_embedding_config()
    client = manager or get_default_gemini_manager()

    embeddings: list[list[float]] = []
    for batch in _batch(texts, cfg.batch_size):
        embeddings.extend(client.embed_documents(batch))

    return embeddings


def embed_query(
    text: str,
    manager: GeminiClientManager | None = None,
) -> list[float]:
    """Embed a single query text for retrieval."""

    client = manager or get_default_gemini_manager()
    vector = client.embed_query(text)
    if not vector:
        raise RuntimeError("Embedding API returned no vector.")
    return vector
