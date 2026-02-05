# clean-normalize/table_normalization.py

from typing import List
from schema.ingestion import ContentBlock, ContentType


def table_normalization(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """
    Phase-2.4: Table Normalization

    - Works only on TABLE blocks
    - Preserves row/column structure
    - Annotates metadata with normalized table info
    - Does NOT flatten into text
    """

    for block in blocks:
        if block.content_type != ContentType.TABLE:
            continue

        raw = block.text.strip() if block.text else ""
        if not raw:
            block.metadata.extra["empty_table"] = True
            continue

        # Basic delimiter detection
        if "|" in raw:
            delimiter = "|"
        elif "\t" in raw:
            delimiter = "\t"
        else:
            delimiter = None

        rows = []
        if delimiter:
            for line in raw.splitlines():
                cells = [cell.strip() for cell in line.split(delimiter)]
                if any(cells):
                    rows.append(cells)

        # Fallback: newline-separated pseudo-rows
        if not rows:
            rows = [[line.strip()] for line in raw.splitlines() if line.strip()]

        # Header detection (very conservative)
        header = rows[0] if len(rows) > 1 else None
        data_rows = rows[1:] if header else rows

        # Metadata annotation (NO mutation of text)
        block.metadata.extra.update(
            {
                "table_normalized": True,
                "table_rows": len(data_rows),
                "table_columns": max(len(r) for r in rows),
                "table_has_header": header is not None,
            }
        )

    return blocks