"""Document indexing and upsert pipeline for the vector store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from src.core.pinecone_client import get_pinecone_index
from src.processing.chunking import chunk_pdf
from src.processing.drive_download import download_drive_folder_pdfs, download_drive_pdf
from src.rag.embedder import embed_texts


def index_pdf_from_drive(
    *,
    drive_url: str,
    index_name: str,
    local_dir: str | Path = "data/cache",
    similarity_option: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Download a PDF from Google Drive, chunk it, embed it, and upsert to Pinecone.
    """

    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    local_pdf_path = local_dir_path / "drive_download.pdf"

    downloaded_path = download_drive_pdf(drive_url, local_pdf_path)
    chunks = chunk_pdf(downloaded_path)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    if not embeddings:
        raise RuntimeError("No embeddings generated for PDF chunks.")

    vector_dim = len(embeddings[0])
    index = get_pinecone_index(
        index_name,
        vector_dim=vector_dim,
        similarity_option=similarity_option,
        cloud=cloud,
        region=region,
    )

    vectors = []
    for chunk, vector in zip(chunks, embeddings, strict=False):
        vec_meta = {"section": chunk.get("section", ""), "text": chunk.get("text", "")}
        if metadata:
            vec_meta.update(metadata)

        vectors.append(
            {
                "id": chunk["id"],
                "values": vector,
                "metadata": vec_meta,
            }
        )

    upsert_result = index.upsert(vectors=vectors)
    return {
        "downloaded_path": str(downloaded_path),
        "chunks": len(chunks),
        "upserted": len(vectors),
        "result": upsert_result,
    }


def index_drive_folder(
    *,
    drive_folder_url: str,
    index_name: str,
    local_dir: str | Path = "data/cache",
    similarity_option: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    metadata: dict[str, Any] | None = None,
    upsert_batch_size: int = 200,
) -> dict[str, Any]:
    """
    Download all PDFs from a Google Drive folder, chunk them, embed them, and upsert to Pinecone.
    """

    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    pdf_paths = download_drive_folder_pdfs(drive_folder_url, local_dir_path)

    total_chunks = 0
    total_upserted = 0
    upsert_result = None

    index = None
    for pdf_path in pdf_paths:
        chunks = chunk_pdf(pdf_path)
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        if not embeddings:
            raise RuntimeError(f"No embeddings generated for {pdf_path}.")

        if index is None:
            vector_dim = len(embeddings[0])
            index = get_pinecone_index(
                index_name,
                vector_dim=vector_dim,
                similarity_option=similarity_option,
                cloud=cloud,
                region=region,
            )

        vectors = []
        file_stem = pdf_path.stem
        for chunk, vector in zip(chunks, embeddings, strict=False):
            vec_meta = {
                "section": chunk.get("section", ""),
                "text": chunk.get("text", ""),
                "source_file": pdf_path.name,
            }
            if metadata:
                vec_meta.update(metadata)

            vectors.append(
                {
                    "id": f"{file_stem}__{chunk['id']}",
                    "values": vector,
                    "metadata": vec_meta,
                }
            )

        for batch in _batched(vectors, upsert_batch_size):
            upsert_result = index.upsert(vectors=batch)
            total_upserted += len(batch)

        total_chunks += len(chunks)

    return {
        "downloaded": len(pdf_paths),
        "chunks": total_chunks,
        "upserted": total_upserted,
        "result": upsert_result,
    }


def index_local_folder(
    *,
    local_folder: str | Path,
    index_name: str,
    similarity_option: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
    metadata: dict[str, Any] | None = None,
    upsert_batch_size: int = 200,
) -> dict[str, Any]:
    """Index all PDFs from a local folder into Pinecone."""

    folder = Path(local_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Local folder not found: {folder}")

    pdf_paths = sorted(p for p in folder.rglob("*.pdf") if p.is_file())
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in local folder: {folder}")

    total_chunks = 0
    total_upserted = 0
    upsert_result = None

    index = None
    for pdf_path in pdf_paths:
        chunks = chunk_pdf(pdf_path)
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        if not embeddings:
            raise RuntimeError(f"No embeddings generated for {pdf_path}.")

        if index is None:
            vector_dim = len(embeddings[0])
            index = get_pinecone_index(
                index_name,
                vector_dim=vector_dim,
                similarity_option=similarity_option,
                cloud=cloud,
                region=region,
            )

        vectors = []
        file_stem = pdf_path.stem
        for chunk, vector in zip(chunks, embeddings, strict=False):
            vec_meta = {
                "section": chunk.get("section", ""),
                "text": chunk.get("text", ""),
                "source_file": pdf_path.name,
            }
            if metadata:
                vec_meta.update(metadata)

            vectors.append(
                {
                    "id": f"{file_stem}__{chunk['id']}",
                    "values": vector,
                    "metadata": vec_meta,
                }
            )

        for batch in _batched(vectors, upsert_batch_size):
            upsert_result = index.upsert(vectors=batch)
            total_upserted += len(batch)

        total_chunks += len(chunks)

    return {
        "downloaded": len(pdf_paths),
        "chunks": total_chunks,
        "upserted": total_upserted,
        "result": upsert_result,
    }


def index_all_from_env(
    *,
    local_dir: str | Path = "data/cache",
    similarity_option: str = "cosine",
    cloud: str | None = None,
    region: str | None = None,
    index_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Index all configured Drive folders into their respective Pinecone indexes."""

    cloud = cloud or os.getenv("PINECONE_CLOUD", "aws").strip()
    region = region or os.getenv("PINECONE_REGION", "us-east-1").strip()

    mapping = {
        "rta_act": (
            os.getenv("INDEX_SPPA_ACT", ""),
            os.getenv("INDEX_RTA_ACT_DRIVE_URL", ""),
            os.getenv("INDEX_RTA_ACT_LOCAL_DIR", ""),
        ),
        "sppa_act": (
            os.getenv("INDEX_SPPA_ACT", ""),
            os.getenv("INDEX_SPPA_ACT_DRIVE_URL", ""),
            os.getenv("INDEX_SPPA_ACT_LOCAL_DIR", ""),
        ),
        "ltb_rules": (
            os.getenv("INDEX_LTB_RULES", ""),
            os.getenv("INDEX_LTB_RULES_DRIVE_URL", ""),
            os.getenv("INDEX_LTB_RULES_LOCAL_DIR", ""),
        ),
        "ltb_practice_directions": (
            os.getenv("INDEX_LTB_PRACTICE_DIRECTIONS", ""),
            os.getenv("INDEX_LTB_PRACTICE_DIRECTIONS_DRIVE_URL", ""),
            os.getenv("INDEX_LTB_PRACTICE_DIRECTIONS_LOCAL_DIR", ""),
        ),
        "ltb_guidelines": (
            os.getenv("INDEX_LTB_GUIDELINES", ""),
            os.getenv("INDEX_LTB_GUIDELINES_DRIVE_URL", ""),
            os.getenv("INDEX_LTB_GUIDELINES_LOCAL_DIR", ""),
        ),
        "form_instructions": (
            os.getenv("INDEX_FORM_INSTRUCTIONS", ""),
            os.getenv("INDEX_FORM_INSTRUCTIONS_DRIVE_URL", ""),
            os.getenv("INDEX_FORM_INSTRUCTIONS_LOCAL_DIR", ""),
        ),
    }

    allowed = set(index_keys) if index_keys else None

    results: dict[str, Any] = {}
    for key, (index_name, drive_url, local_dir_override) in mapping.items():
        if allowed is not None and key not in allowed:
            continue
        index_name = (index_name or "").strip()
        drive_url = (drive_url or "").strip()
        local_dir_override = (local_dir_override or "").strip()
        if not index_name:
            results[key] = {"skipped": True, "reason": "missing index name"}
            continue

        local_default_dir = Path(local_dir) / key
        if not local_dir_override and local_default_dir.exists():
            local_dir_override = str(local_default_dir)

        if local_dir_override:
            results[key] = index_local_folder(
                local_folder=local_dir_override,
                index_name=index_name,
                similarity_option=similarity_option,
                cloud=cloud,
                region=region,
                metadata={"index_key": key, "local_dir": local_dir_override},
            )
        else:
            if not drive_url:
                results[key] = {"skipped": True, "reason": "missing drive url"}
                continue
            results[key] = index_drive_folder(
                drive_folder_url=drive_url,
                index_name=index_name,
                local_dir=Path(local_dir) / key,
                similarity_option=similarity_option,
                cloud=cloud,
                region=region,
                metadata={"index_key": key, "drive_url": drive_url},
            )

    return results


def _batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]
