"""ChromaDB HTTP client helpers for consultant RAG.

This avoids importing the chromadb Python package inside the API container,
which can be incompatible with Python 3.14 in some environments.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import httpx


def _base_url() -> str:
    host = os.getenv("CHROMA_HOST", "chroma")
    port = os.getenv("CHROMA_PORT", "8000")
    return f"http://{host}:{port}"


def _base_path() -> str:
    tenant = os.getenv("CHROMA_TENANT", "default_tenant")
    database = os.getenv("CHROMA_DATABASE", "default_database")
    return f"{_base_url()}/api/v2/tenants/{tenant}/databases/{database}"


def _extract_collections(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("collections"), list):
            return payload["collections"]
        if isinstance(payload.get("data"), list):
            return payload["data"]
    return []


def get_or_create_collection(name: str | None = None) -> Tuple[str, str]:
    """Ensure a Chroma collection exists and return (collection_id, name)."""

    collection_name = name or os.getenv("CHROMA_COLLECTION", "canlii_cases")
    base = _base_path()
    url = f"{base}/collections"

    with httpx.Client(timeout=30) as client:
        # 1) Try to find existing collection by listing
        list_resp = client.get(url)
        if list_resp.status_code == 200:
            collections = _extract_collections(list_resp.json())
            for item in collections:
                if item.get("name") == collection_name:
                    return item.get("id", collection_name), collection_name

        # Try direct GET by name (some deployments support this)
        # No direct GET by name in v2; rely on listing above.

        # 2) Try to create collection
        create_resp = client.post(url, json={"name": collection_name})
        if create_resp.status_code in (200, 201):
            data = create_resp.json() or {}
            return data.get("id", collection_name), collection_name

        # 3) Some Chroma builds may not allow POST on /collections (405)
        if create_resp.status_code == 405:
            return collection_name, collection_name

        create_resp.raise_for_status()

    return collection_name, collection_name


def chroma_add(
    collection_id: str,
    *,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    url = f"{_base_path()}/collections/{collection_id}/add"
    payload = {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas,
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()


def chroma_query(
    collection_id: str,
    *,
    query_embeddings: list[list[float]],
    n_results: int = 10,
) -> dict:
    payload = {"query_embeddings": query_embeddings, "n_results": n_results}
    urls = [
        f"{_base_path()}/collections/{collection_id}/query",
        f"{_base_path()}/collections/{collection_id}/query/",
    ]
    with httpx.Client(timeout=60) as client:
        last_error = None
        for url in urls:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json()
            last_error = resp
        if last_error is not None:
            last_error.raise_for_status()
    return {}
