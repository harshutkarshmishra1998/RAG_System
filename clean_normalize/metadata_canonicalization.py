# clean-normalize/metadata_canonicalization.py

from typing import List
from schema.ingestion import ContentBlock, ContentType, SourceType


def metadata_canonicalization(
    blocks: List[ContentBlock],
    source_type: SourceType,
) -> List[ContentBlock]:
    """
    Phase-2.5: Metadata Canonicalization

    Normalizes block metadata into predictable keys.
    No deletion. No overwriting of source data.
    """

    for block in blocks:
        meta = block.metadata.extra

        # --------------------------------------------------
        # Canonical content markers
        # --------------------------------------------------
        meta.setdefault("content_type", block.content_type.value)

        # --------------------------------------------------
        # OCR awareness
        # --------------------------------------------------
        meta.setdefault(
            "is_ocr",
            block.content_type == ContentType.IMAGE_OCR,
        )

        # --------------------------------------------------
        # Table awareness
        # --------------------------------------------------
        meta.setdefault(
            "has_table",
            block.content_type == ContentType.TABLE,
        )

        # --------------------------------------------------
        # Source awareness
        # --------------------------------------------------
        meta.setdefault("source_type", source_type.value)

        # --------------------------------------------------
        # Structural flags (defaults)
        # --------------------------------------------------
        meta.setdefault("empty_block", False)
        meta.setdefault("noise_block", False)
        meta.setdefault("boilerplate", False)

        # --------------------------------------------------
        # Confidence (only set if not provided)
        # --------------------------------------------------
        meta.setdefault("confidence", None)

    return blocks
