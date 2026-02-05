# clean-normalize/structural_block_segmentation.py

from typing import List
from schema.ingestion import ContentBlock, ContentType


def structural_block_segmentation(
    blocks: List[ContentBlock],
) -> List[ContentBlock]:
    """
    Phase-2.2: Structural Block Segmentation

    - Flags empty blocks
    - Flags content_type ↔ text mismatches
    - Marks obvious noise
    - Does NOT delete or reorder blocks
    """
    for block in blocks:
        text = block.text.strip() if block.text else ""

        # --------------------------------------------------
        # Empty block detection
        # --------------------------------------------------
        if not text:
            block.metadata.extra["empty_block"] = True
            continue

        # --------------------------------------------------
        # Content-type vs text sanity checks
        # --------------------------------------------------
        if block.content_type == ContentType.TABLE:
            if "\n" not in text and "|" not in text:
                block.metadata.extra["table_malformed"] = True

        if block.content_type == ContentType.IMAGE_OCR:
            # OCR noise heuristic (very conservative)
            alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.3:
                block.metadata.extra["ocr_noise"] = True

        # --------------------------------------------------
        # Ultra-short noise blocks
        # --------------------------------------------------
        if len(text) < 3:
            block.metadata.extra["noise_block"] = True

    return blocks