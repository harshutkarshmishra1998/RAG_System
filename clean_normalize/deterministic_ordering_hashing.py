# clean-normalize/deterministic_ordering_hashing.py

import hashlib
from typing import List
from schema.ingestion import ContentBlock


def _stable_text(text: str) -> str:
    """
    Normalize text for hashing without changing semantics.
    """
    return text.strip().lower() if text else ""


def _hash_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deterministic_ordering_and_hashing(
    blocks: List[ContentBlock],
) -> List[ContentBlock]:
    """
    Phase-2.6: Deterministic Ordering & Hashing

    - Computes stable content hashes
    - Annotates metadata.extra with hashes
    - Enforces stable ordering via hash keys
    """

    for block in blocks:
        stable_text = _stable_text(block.text)

        # Content hash (stable across runs)
        content_hash = _hash_string(stable_text)

        # Structural hash (text + metadata signals)
        structural_seed = (
            stable_text
            + str(block.content_type.value)
            + str(block.metadata.page.page_number if block.metadata.page else "")
        )
        structural_hash = _hash_string(structural_seed)

        block.metadata.extra.setdefault("content_hash", content_hash)
        block.metadata.extra.setdefault("structural_hash", structural_hash)

    # Stable ordering (ONLY if needed)
    blocks.sort(
        key=lambda b: (
            b.metadata.page.page_number
            if b.metadata.page and b.metadata.page.page_number is not None
            else -1,
            b.metadata.extra.get("structural_hash"),
        )
    )

    return blocks
