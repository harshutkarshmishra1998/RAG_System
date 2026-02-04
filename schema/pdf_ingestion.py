from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from schema.ingestion import (
    IngestedDocument,
    ContentBlock,
    ContentType,
    SourceMetadata,
    SourceType,
    BlockMetadata,
    PageMetadata,
)

# ============================================================
# Runtime dependency resolution (executed at import time)
# Assumes `.env` has already been loaded by caller
# ============================================================

def _resolve_poppler_path() -> str | None:
    """
    Resolve Poppler binaries.

    Priority:
    1. POPPLER_PATH env var (directory)
    2. PATH lookup (pdftoppm)

    Returns:
        Directory path or None (use PATH)
    """
    poppler_path = os.getenv("POPPLER_PATH")

    if poppler_path:
        if not os.path.isdir(poppler_path):
            raise RuntimeError(f"Invalid POPPLER_PATH: {poppler_path}")
        return poppler_path

    if shutil.which("pdftoppm"):
        return None

    raise RuntimeError(
        "Poppler not found. Set POPPLER_PATH or add Poppler to PATH."
    )


def _resolve_tesseract() -> None:
    """
    Resolve Tesseract executable.

    Priority:
    1. TESSERACT_PATH env var (full path)
    2. PATH lookup
    """
    tesseract_path = os.getenv("TESSERACT_PATH")

    if tesseract_path:
        if not os.path.isfile(tesseract_path):
            raise RuntimeError(f"Invalid TESSERACT_PATH: {tesseract_path}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return

    if shutil.which("tesseract"):
        return

    raise RuntimeError(
        "Tesseract not found. Set TESSERACT_PATH or add it to PATH."
    )


# Resolve once (fail fast)
POPPLER_PATH = _resolve_poppler_path()
_resolve_tesseract()

# ============================================================
# PDF ingestion
# ============================================================

def ingest_pdf(pdf_path: str | Path) -> IngestedDocument:
    """
    Ingest a PDF into an IngestedDocument.

    Steps:
    1. Extract native text
    2. Extract tables
    3. OCR image-only pages
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    blocks: List[ContentBlock] = []

    # --------------------------------------------------------
    # Pass 1: Native text + tables
    # --------------------------------------------------------
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text()
            if text and text.strip():
                blocks.append(
                    ContentBlock(
                        content_type=ContentType.TEXT,
                        text=text.strip(),
                        metadata=BlockMetadata(
                            page=PageMetadata(page_number=page_number)
                        ),
                    )
                )

            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(
                    ["\t".join(cell or "" for cell in row) for row in table]
                )

                if table_text.strip():
                    blocks.append(
                        ContentBlock(
                            content_type=ContentType.TABLE,
                            text=table_text,
                            metadata=BlockMetadata(
                                page=PageMetadata(page_number=page_number)
                            ),
                        )
                    )

    # --------------------------------------------------------
    # Pass 2: OCR fallback (only pages without native text)
    # --------------------------------------------------------
    images = convert_from_path(
        pdf_path,
        dpi=200,
        poppler_path=POPPLER_PATH,  # type: ignore
    )

    for page_number, image in enumerate(images, start=1):

        has_text = any(
            block.content_type == ContentType.TEXT
            and block.metadata.page
            and block.metadata.page.page_number == page_number
            for block in blocks
        )

        if has_text:
            continue

        ocr_text = pytesseract.image_to_string(image).strip()

        if ocr_text:
            blocks.append(
                ContentBlock(
                    content_type=ContentType.IMAGE_OCR,
                    text=ocr_text,
                    metadata=BlockMetadata(
                        page=PageMetadata(page_number=page_number),
                        extra={"method": "tesseract"},
                    ),
                )
            )

    return IngestedDocument(
        source=SourceMetadata(
            source_type=SourceType.PDF,
            source_uri=str(pdf_path),
            file_name=pdf_path.name,
        ),
        blocks=blocks,
    )
