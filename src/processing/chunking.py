"""Chunking utilities for RAG indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .pdf_to_text import convert_pdf_to_text

try:
    from langchain_core.tools import tool as lc_tool
except Exception:  # pragma: no cover - optional dependency for tool decorator
    lc_tool = None


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    min_chunk_tokens: int
    heading_prefixes: list[str]
    separators: list[str]


DEFAULT_CONFIG_PATH = Path("config/chunking_config.yaml")


def load_chunking_config(path: str | Path | None = None) -> ChunkingConfig:
    """Load chunking configuration from YAML."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Chunking config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}

    return ChunkingConfig(
        chunk_size_tokens=int(raw.get("chunk_size_tokens", 800)),
        chunk_overlap_tokens=int(raw.get("chunk_overlap_tokens", 120)),
        min_chunk_tokens=int(raw.get("min_chunk_tokens", 200)),
        heading_prefixes=[s.lower() for s in raw.get("heading_prefixes", [])],
        separators=raw.get("separators", ["\n\n", "\n", " ", ""]),
    )


def _token_count(text: str) -> int:
    """Approximate token count using whitespace splitting."""

    return len(text.split())


def _split_by_headings(text: str, heading_prefixes: list[str]) -> list[tuple[str, str]]:
    """Split text into sections based on heading prefixes."""

    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        lower = stripped.lower()
        is_heading = any(lower.startswith(prefix) for prefix in heading_prefixes)

        if is_heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _recursive_split(text: str, max_tokens: int, separators: list[str]) -> list[str]:
    """Recursively split text to fit within max token size."""

    if _token_count(text) <= max_tokens:
        return [text]

    if not separators:
        words = text.split()
        chunks: list[str] = []
        for i in range(0, len(words), max_tokens):
            chunks.append(" ".join(words[i : i + max_tokens]))
        return chunks

    sep = separators[0]
    if sep == "":
        return _recursive_split(text, max_tokens, separators[1:])

    parts = text.split(sep)
    chunks: list[str] = []
    buffer: list[str] = []

    for part in parts:
        candidate = (sep.join(buffer + [part])).strip()
        if not candidate:
            continue
        if _token_count(candidate) <= max_tokens:
            buffer.append(part)
            continue

        if buffer:
            chunks.extend(_recursive_split(sep.join(buffer).strip(), max_tokens, separators[1:]))
            buffer = [part]
        else:
            chunks.extend(_recursive_split(part.strip(), max_tokens, separators[1:]))

    if buffer:
        chunks.extend(_recursive_split(sep.join(buffer).strip(), max_tokens, separators[1:]))

    return [c for c in chunks if c]


def _apply_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Apply sliding overlap between adjacent chunks."""

    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    prev_words: list[str] = []

    for chunk in chunks:
        words = chunk.split()
        if prev_words:
            prefix = " ".join(prev_words[-overlap_tokens:])
            chunk = f"{prefix} {chunk}".strip()
        overlapped.append(chunk)
        prev_words = words

    return overlapped


def chunk_text(text: str, config_path: str | Path | None = None) -> list[dict[str, str]]:
    """Chunk text using document-based headings + recursive splitting."""

    cfg = load_chunking_config(config_path)

    sections = _split_by_headings(text, cfg.heading_prefixes)
    if not sections:
        sections = [("", text)]

    results: list[dict[str, str]] = []
    chunk_id = 0

    for title, body in sections:
        if not body:
            continue
        chunks = _recursive_split(body, cfg.chunk_size_tokens, cfg.separators)
        chunks = _apply_overlap(chunks, cfg.chunk_overlap_tokens)

        for chunk in chunks:
            if _token_count(chunk) < cfg.min_chunk_tokens:
                continue
            results.append(
                {
                    "id": f"chunk_{chunk_id}",
                    "section": title,
                    "text": chunk,
                }
            )
            chunk_id += 1

    return results


def chunk_pdf(pdf_path: str | Path, config_path: str | Path | None = None) -> list[dict[str, str]]:
    """Convert PDF to text and chunk it for indexing."""

    text = convert_pdf_to_text(pdf_path)
    return chunk_text(text, config_path=config_path)


if lc_tool is not None:

    @lc_tool("chunk_text")
    def chunk_text_tool(text: str) -> list[dict[str, str]]:  # pragma: no cover
        """Chunk text for RAG indexing using configured settings."""

        return chunk_text(text)

    @lc_tool("chunk_pdf")
    def chunk_pdf_tool(pdf_path: str) -> list[dict[str, str]]:  # pragma: no cover
        """Convert a PDF to text and chunk it for RAG indexing."""

        return chunk_pdf(pdf_path)

