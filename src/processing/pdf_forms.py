"""PDF form field extraction utilities for structured ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def extract_pdf_form_fields(
    pdf_path: str | Path,
    *,
    password: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract AcroForm fields from a PDF.

    Returns a list of dicts with field metadata so downstream systems can
    preserve form inputs, types, and allowed options.
    """

    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path), password=password)
    fields = reader.get_fields() or {}

    results: list[dict[str, Any]] = []
    for name, field in fields.items():
        results.append(
            {
                "name": name,
                "value": field.get("/V"),
                "type": field.get("/FT"),
                "flags": field.get("/Ff"),
                "options": field.get("/Opt"),
                "alternate_name": field.get("/TU"),
                "mapping_name": field.get("/TM"),
            }
        )

    return results


def extract_pdf_form_fields_tool(
    pdf_path: str,
    password: str | None = None,
) -> list[dict[str, Any]]:
    """
    Tool-friendly wrapper for extracting PDF form fields.
    """

    return extract_pdf_form_fields(pdf_path, password=password)
