"""Crawler and indexer for CanLII LTB cases."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

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
    max_pages: int = 50,
    sleep_seconds: float = 0.25,
) -> list[str]:
    """Crawl CanLII listing pages and collect case page URLs."""

    visited = set()
    to_visit = [start_url]
    case_pages: set[str] = set()

    with httpx.Client(timeout=30) as client:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            resp = client.get(url, headers={"User-Agent": "mll-crawler/1.0"})
            resp.raise_for_status()

            links = _extract_links(resp.text, url)
            for link in links:
                if not _is_same_domain(link, start_url):
                    continue

                # Case pages usually contain '/doc/'
                if "/doc/" in link:
                    case_pages.add(link)
                # Follow pagination or navigation
                if "/nav/" in link or "?" in link:
                    if link not in visited:
                        to_visit.append(link)

            time.sleep(sleep_seconds)

    return sorted(case_pages)


def fetch_case_text(url: str) -> str:
    """Fetch and extract text from a CanLII case page."""

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers={"User-Agent": "mll-crawler/1.0"})
        resp.raise_for_status()
        return _extract_text_from_html(resp.text)


def find_pdf_links(url: str) -> list[str]:
    """Find PDF links within a case page."""

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers={"User-Agent": "mll-crawler/1.0"})
        resp.raise_for_status()
        links = _extract_links(resp.text, url)

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
    max_pages: int = 30,
) -> dict:
    """End-to-end crawler + indexer for CanLII cases."""

    case_pages = crawl_canlii_case_pages(start_url=start_url, max_pages=max_pages)

    documents = []
    for page in case_pages:
        text = fetch_case_text(page)
        documents.append({"text": text, "metadata": {"source": page}})

        # Optional: find and parse PDFs linked from the case page
        for pdf_link in find_pdf_links(page):
            pdf_path = download_pdf(pdf_link)
            pdf_text = convert_pdf_to_text(pdf_path)
            documents.append(
                {"text": pdf_text, "metadata": {"source": pdf_link, "type": "pdf"}}
            )

    return index_case_documents(documents)
