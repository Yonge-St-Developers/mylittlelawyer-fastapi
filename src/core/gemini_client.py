"""Gemini API client wrapper for LLM calls."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from google import genai
from google.genai import errors as genai_errors


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
        response = self._retry_generate_content(
            model=self._config.model_name,
            contents=contents,
            **kwargs,
        )
        return response.text or ""

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        response = self._retry_embed_content(
            model=self._config.embedding_model,
            contents=list(texts),
        )
        embeddings = response.embeddings or []
        return [e.values for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval."""
        response = self._retry_embed_content(
            model=self._config.embedding_model,
            contents=text,
        )
        embeddings = response.embeddings or []
        if not embeddings:
            return []
        return embeddings[0].values

    def _retry_generate_content(self, **kwargs: Any):
        """Retry wrapper for generate_content (handles 429/503)."""

        return _retry_with_backoff(self._client.models.generate_content, **kwargs)

    def _retry_embed_content(self, **kwargs: Any):
        """Retry wrapper for embed_content (handles 429/503)."""

        return _retry_with_backoff(self._client.models.embed_content, **kwargs)


def _retry_with_backoff(fn, **kwargs: Any):
    """Basic retry with exponential backoff for transient Gemini errors."""

    max_attempts = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    base_delay = float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "2"))

    attempt = 0
    while True:
        try:
            return fn(**kwargs)
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise

            # If API provides retryDelay in error details, honor it
            retry_delay = base_delay * (2 ** (attempt - 1))
            try:
                details = getattr(exc, "details", None) or []
                for item in details:
                    if isinstance(item, dict) and item.get("@type", "").endswith("RetryInfo"):
                        delay = item.get("retryDelay", "")
                        if isinstance(delay, str) and delay.endswith("s"):
                            retry_delay = float(delay[:-1])
            except Exception:
                pass

            time.sleep(retry_delay)


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
