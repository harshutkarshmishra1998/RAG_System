from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin

from schema.ingestion import (
    IngestedDocument,
    ContentBlock,
    ContentType,
    SourceMetadata,
    SourceType,
    BlockMetadata,
)


# ============================================================
# Web page ingestion (text + tables + images)
# ============================================================

def ingest_web_page(url: str) -> IngestedDocument:
    """
    Ingest a public web page.

    Extracts:
    - Text blocks (paragraphs, headings, lists)
    - Tables (HTML tables → serialized text)
    - Images (metadata only; OCR optional later)
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
    }

    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # --------------------------------------------------------
    # Remove non-content / noisy tags
    # --------------------------------------------------------
    for tag in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "aside"]
    ):
        tag.decompose()

    blocks: List[ContentBlock] = []

    # --------------------------------------------------------
    # 1️⃣ TEXT (paragraphs, headings, list items)
    # --------------------------------------------------------
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        text = elem.get_text(separator=" ", strip=True)
        if not text or len(text) < 5:
            continue

        blocks.append(
            ContentBlock(
                content_type=ContentType.TEXT,
                text=text,
                metadata=BlockMetadata(
                    extra={"html_tag": elem.name} #type: ignore
                ),
            )
        )

    # --------------------------------------------------------
    # 2️⃣ TABLES
    # --------------------------------------------------------
    for table in soup.find_all("table"):
        rows = []

        for tr in table.find_all("tr"): #type: ignore
            cells = [
                cell.get_text(separator=" ", strip=True)
                for cell in tr.find_all(["th", "td"]) #type: ignore
            ]
            if cells:
                rows.append(" | ".join(cells))

        if not rows:
            continue

        table_text = "\n".join(rows)

        blocks.append(
            ContentBlock(
                content_type=ContentType.TABLE,
                text=table_text,
                metadata=BlockMetadata(
                    extra={"rows": len(rows)}
                ),
            )
        )

    # --------------------------------------------------------
    # 3️⃣ IMAGES (metadata-only for Phase 1)
    # --------------------------------------------------------
    for img in soup.find_all("img"):
        src = img.get("src") #type: ignore
        if not src:
            continue

        img_url = urljoin(url, src) #type: ignore

        alt_text = img.get("alt", "").strip() #type: ignore

        blocks.append(
            ContentBlock(
                content_type=ContentType.IMAGE_OCR,
                text=alt_text if alt_text else "",
                metadata=BlockMetadata(
                    extra={
                        "image_url": img_url,
                        "alt": alt_text,
                        "ocr_pending": True,
                    }
                ),
            )
        )


    if not blocks:
        raise RuntimeError("No extractable content found on web page")

    return IngestedDocument(
        source=SourceMetadata(
            source_type=SourceType.WEB,
            source_uri=url,
            file_name=None,
        ),
        blocks=blocks,
    )