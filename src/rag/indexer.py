"""Document indexing and upsert pipeline for the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.pinecone_client import get_pinecone_index
from src.processing.chunking import chunk_pdf
from src.processing.drive_download import download_drive_pdf
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
