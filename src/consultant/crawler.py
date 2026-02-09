"""Crawler and indexer for CanLII LTB cases."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

from src.consultant.chroma_client import chroma_add, get_or_create_collection
from src.processing.chunking import chunk_text
from src.processing.pdf_to_text import convert_pdf_to_text
from src.rag.embedder import embed_texts

DEFAULT_START_URL = "https://www.canlii.org/on/onltb/nav/date/2026"


def _is_same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Try common CanLII content container first
    main = soup.find(id="document") or soup.find("div", class_="document")
    if main:
        return _clean_text(main.get_text(" "))

    return _clean_text(soup.get_text(" "))


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        links.append(abs_url)
    return links


def crawl_canlii_case_pages(
    start_url: str = DEFAULT_START_URL,
    max_depth: int = 3,
    max_pages: int = 6,
) -> list[dict]:
    """Crawl CanLII pages using LangChain RecursiveUrlLoader."""

    def _html_extractor(html: str) -> str:
        # Keep raw HTML for link extraction; text is derived later.
        return html

    loader = RecursiveUrlLoader(
        start_url,
        max_depth=max_depth,
        prevent_outside=True,
        extractor=_html_extractor,
    )

    docs = loader.load()
    docs = docs[:max_pages]

    results = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        html = doc.page_content or ""
        results.append({"source": source, "html": html})

    return results


def fetch_case_text_from_html(html: str) -> str:
    """Extract text from a CanLII page's HTML."""

    return _extract_text_from_html(html)


def find_pdf_links_from_html(html: str, base_url: str) -> list[str]:
    """Find PDF links within a page's HTML."""

    links = _extract_links(html, base_url)
    return [link for link in links if link.lower().endswith(".pdf")]


def download_pdf(url: str, dest_dir: str | Path = "data/cache") -> Path:
    """Download a PDF to a local path and return the file path."""

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(urlparse(url).path).name or "case.pdf"
    out_path = dest_dir / file_name

    with httpx.stream("GET", url, timeout=60) as response:
        response.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    return out_path


def index_case_documents(
    documents: Iterable[dict],
    collection_name: str | None = None,
) -> dict:
    """Chunk, embed, and upsert case documents into ChromaDB."""

    collection_id, _ = get_or_create_collection(collection_name)

    texts = [doc["text"] for doc in documents if doc.get("text")]
    metadatas = [doc.get("metadata", {}) for doc in documents if doc.get("text")]

    chunks = []
    chunk_meta = []
    for text, meta in zip(texts, metadatas, strict=False):
        for chunk in chunk_text(text):
            chunks.append(chunk["text"])
            chunk_meta.append(meta)

    if not chunks:
        return {"indexed": 0}

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


def crawl_and_index(
    start_url: str = DEFAULT_START_URL,
    max_pages: int = 6,
    max_depth: int = 3,
) -> dict:
    """End-to-end crawler + indexer for CanLII cases."""

    pages = crawl_canlii_case_pages(
        start_url=start_url, max_pages=max_pages, max_depth=max_depth
    )

    documents = []
    for page in pages:
        source = page["source"]
        html = page["html"]
        text = fetch_case_text_from_html(html)
        documents.append({"text": text, "metadata": {"source": source}})

        # Optional: find and parse PDFs linked from the case page
        for pdf_link in find_pdf_links_from_html(html, source):
            pdf_path = download_pdf(pdf_link)
            pdf_text = convert_pdf_to_text(pdf_path)
            documents.append(
                {"text": pdf_text, "metadata": {"source": pdf_link, "type": "pdf"}}
            )

    return index_case_documents(documents)
