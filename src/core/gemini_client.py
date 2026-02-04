"""Gemini API client wrapper for LLM calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from google import genai


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for Gemini API usage."""

    api_key: str
    model_name: str
    embedding_model: str


class GeminiClientManager:
    """Centralized Gemini client and model manager."""

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        self._client = genai.Client(api_key=config.api_key)

    @property
    def config(self) -> GeminiConfig:
        return self._config

    def generate_text(self, contents: str, **kwargs: Any) -> str:
        """Generate text with the configured Gemini model."""

        response = self._client.models.generate_content(
            model=self._config.model_name,
            contents=contents,
            **kwargs,
        )
        return response.text or ""

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of documents."""

        response = self._client.models.embed_content(
            model=self._config.embedding_model,
            contents=list(texts),
            task_type="retrieval_document",
        )
        embeddings = response.embeddings or []
        return [e.values for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval."""

        response = self._client.models.embed_content(
            model=self._config.embedding_model,
            contents=text,
            task_type="retrieval_query",
        )
        embeddings = response.embeddings or []
        if not embeddings:
            return []
        return embeddings[0].values


class GeminiEmbeddingClient:
    """Dedicated embedding client using the Gemini embeddings API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = genai.Client(api_key=api_key)

    def embed_content(self, model: str, contents: str | list[str]) -> list[list[float]]:
        """Call the embeddings API and return vectors."""

        result = self._client.models.embed_content(
            model=model,
            contents=contents,
        )
        embeddings = result.embeddings or []
        return [e.values for e in embeddings]

def _load_config_from_env() -> GeminiConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "").strip()
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    if not model_name:
        raise ValueError("GEMINI_MODEL is not set.")
    if not embedding_model:
        raise ValueError("GEMINI_EMBEDDING_MODEL is not set.")

    return GeminiConfig(
        api_key=api_key,
        model_name=model_name,
        embedding_model=embedding_model,
    )


@lru_cache(maxsize=1)
def get_default_gemini_manager() -> GeminiClientManager:
    """Get a cached Gemini client manager initialized from environment variables."""

    return GeminiClientManager(_load_config_from_env())
