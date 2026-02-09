"""Local PDF indexer for consultant RAG (no crawling)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.consultant.chroma_client import chroma_add, get_or_create_collection
from src.processing.chunking import chunk_text
from src.processing.pdf_to_text import convert_pdf_to_text
from src.rag.embedder import embed_texts

DEFAULT_PDF_DIR = Path("data/cache/ltb_law_case")


def _iter_pdfs(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.glob("**/*.pdf") if p.is_file())


def index_local_pdfs(folder: Path | str = DEFAULT_PDF_DIR) -> dict:
    """Read PDFs from a local folder, embed, and store in ChromaDB."""

    folder = Path(folder)
    if not folder.exists():
        return {"indexed": 0, "reason": f"Folder not found: {folder}"}

    collection_id, _ = get_or_create_collection()

    documents = []
    for pdf_path in _iter_pdfs(folder):
        text = convert_pdf_to_text(pdf_path)
        if text:
            documents.append({"text": text, "metadata": {"source": str(pdf_path)}})

    if not documents:
        return {"indexed": 0, "reason": "No PDFs found"}

    chunks = []
    chunk_meta = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append(chunk["text"])
            chunk_meta.append(doc["metadata"])

    if not chunks:
        return {"indexed": 0, "reason": "No chunks produced"}

    embeddings = embed_texts(chunks)
    ids = [f"case_{i}" for i in range(len(chunks))]

    chroma_add(
        collection_id,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=chunk_meta,
    )

    return {"indexed": len(chunks)}
