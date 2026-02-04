from __future__ import annotations

from pathlib import Path
from typing import List

from schema.ingestion import (
    IngestedDocument,
    ContentBlock,
    ContentType,
    SourceMetadata,
    SourceType,
    BlockMetadata,
)

# ============================================================
# TXT ingestion
# ============================================================

def ingest_txt(txt_path: str | Path) -> IngestedDocument:
    """
    Ingest a TXT file into an IngestedDocument.

    Strategy:
    - Each non-empty line becomes one TEXT block
    - Whitespace-only lines are ignored
    - No merging or chunking at this phase
    """

    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    blocks: List[ContentBlock] = []

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue

            blocks.append(
                ContentBlock(
                    content_type=ContentType.TEXT,
                    text=text,
                    metadata=BlockMetadata(
                        extra={"line_number": line_number}
                    ),
                )
            )

    return IngestedDocument(
        source=SourceMetadata(
            source_type=SourceType.TXT,
            source_uri=str(txt_path),
            file_name=txt_path.name,
        ),
        blocks=blocks,
    )