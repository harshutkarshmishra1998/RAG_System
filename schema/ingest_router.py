from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Union

from schema.ingestion import IngestedDocument
from schema.pdf_ingestion import ingest_pdf
from schema.docx_ingestion import ingest_docx
from schema.txt_ingestion import ingest_txt
from schema.yt_ingestion import ingest_youtube
from schema.web_ingestion import ingest_web_page
from schema.google_drive_loader import download_from_google_drive


# ============================================================
# URL helpers
# ============================================================

def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _is_youtube_url(value: str) -> bool:
    return bool(re.search(r"(youtube\.com/watch\?v=|youtu\.be/)", value))


def _is_google_drive_or_docs_url(value: str) -> bool:
    return (
        "drive.google.com" in value
        or "docs.google.com/document" in value
    )


# ============================================================
# Unified ingest router
# ============================================================

def ingest(input_value: Union[str, Path]) -> IngestedDocument:
    """
    Unified ingestion entry point.

    Supports:
    - Local files: PDF, DOCX, TXT
    - URLs:
        • YouTube
        • Google Drive files
        • Google Docs (Drive-backed)
        • Generic web pages (HTML)

    Always returns:
        IngestedDocument
    """

    input_str = str(input_value)

    # --------------------------------------------------------
    # Case 1: URL input
    # --------------------------------------------------------
    if _is_url(input_str):

        # ---- YouTube ----
        if _is_youtube_url(input_str):
            return ingest_youtube(input_str)

        # ---- Google Drive OR Google Docs ----
        if _is_google_drive_or_docs_url(input_str):
            with tempfile.TemporaryDirectory() as tmpdir:
                local_file = download_from_google_drive(
                    input_str,
                    Path(tmpdir),
                )
                return ingest(local_file)

        # ---- Generic Web Page ----
        return ingest_web_page(input_str)

    # --------------------------------------------------------
    # Case 2: Local file input
    # --------------------------------------------------------
    path = Path(input_str)

    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return ingest_pdf(path)

    if suffix == ".docx":
        return ingest_docx(path)

    if suffix == ".txt":
        return ingest_txt(path)

    raise ValueError(f"Unsupported file type: {suffix}")