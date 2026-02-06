"""Helpers for downloading PDFs from Google Drive URLs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import httpx
import gdown


_DRIVE_FILE_ID_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_FOLDER_ID_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")


def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract a Google Drive file ID from common URL formats."""

    match = _DRIVE_FILE_ID_RE.search(url)
    if match:
        return match.group(1)

    # Alternate format: https://drive.google.com/open?id=FILE_ID
    if "id=" in url:
        parts = url.split("id=", 1)
        if len(parts) == 2:
            return parts[1].split("&", 1)[0]

    return None


def extract_drive_folder_id(url: str) -> Optional[str]:
    """Extract a Google Drive folder ID from common URL formats."""

    match = _DRIVE_FOLDER_ID_RE.search(url)
    if match:
        return match.group(1)

    if "folders=" in url:
        parts = url.split("folders=", 1)
        if len(parts) == 2:
            return parts[1].split("&", 1)[0]

    return None


def download_drive_folder_pdfs(url: str, dest_dir: str | Path) -> list[Path]:
    """Download all PDFs from a Google Drive folder URL."""

    folder_id = extract_drive_folder_id(url)
    if not folder_id:
        raise ValueError("Could not extract Google Drive folder ID from URL.")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{folder_id}",
        output=str(dest),
        quiet=True,
        use_cookies=False,
    )

    pdfs = sorted(p for p in dest.rglob("*.pdf") if p.is_file())
    if not pdfs:
        raise RuntimeError("No PDF files found after downloading Drive folder.")

    return pdfs


def download_drive_pdf(url: str, dest_path: str | Path) -> Path:
    """Download a PDF from Google Drive and save it locally."""

    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("Could not extract Google Drive file ID from URL.")

    # Use the direct download endpoint
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", download_url, timeout=60.0) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    return dest
