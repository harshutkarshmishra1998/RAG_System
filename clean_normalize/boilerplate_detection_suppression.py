# clean-normalize/boilerplate_detection_suppression.py

from collections import Counter
from typing import List
from schema.ingestion import ContentBlock


def boilerplate_detection_suppression(
    blocks: List[ContentBlock],
    repetition_threshold: int = 3,
    min_length: int = 15,
) -> List[ContentBlock]:
    """
    Phase-2.3: Boilerplate Detection & Suppression

    Heuristics:
    - Short text repeated across blocks → boilerplate
    - Marked, not deleted
    """

    # Normalize text for frequency counting
    normalized_texts = [
        (i, block.text.strip().lower())
        for i, block in enumerate(blocks)
        if block.text and len(block.text.strip()) >= min_length
    ]

    freq = Counter(text for _, text in normalized_texts)

    for idx, text in normalized_texts:
        if freq[text] >= repetition_threshold:
            blocks[idx].metadata.extra["boilerplate"] = True

    return blocks