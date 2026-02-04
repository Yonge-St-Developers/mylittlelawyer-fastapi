from __future__ import annotations
from pathlib import Path
from typing import Iterable
from pypdf import PdfReader


def convert_pdf_to_text(
    pdf_path: str | Path,
    *,
    password: str | None = None,
    max_pages: int | None = None,
) -> str:
    """
    Convert a PDF file to plain text.

    This is designed to be used both by ingestion/indexing pipelines and
    as a callable tool within agentic RAG flows.
    """

    path = Path(pdf_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path), password=password)
    
    pages: Iterable = reader.pages

    if max_pages is not None:
        pages = list(pages)[: max_pages]

    text_chunks: list[str] = []
    for page in pages:
        extracted = page.extract_text() or ""
        text_chunks.append(extracted)

    return "\n".join(text_chunks).strip()


def convert_pdf_to_text_tool(
    pdf_path: str,
    password: str | None = None,
    max_pages: int | None = None,
) -> str:
    """
    Tool-friendly wrapper for PDF to text conversion.

    Keep this signature simple for LLM/tool calling and reuse in database
    ingestion flows.
    """

    return convert_pdf_to_text(pdf_path, password=password, max_pages=max_pages)
