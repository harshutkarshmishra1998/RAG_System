# clean-normalize/canonical_text_sanitation.py

import re
import unicodedata


ZERO_WIDTH_CHARACTERS = (
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # byte order mark
)


def canonical_text_sanitation(text: str) -> str:
    """
    Phase-2.1: Canonical Text Sanitation

    Deterministic, idempotent, semantic-preserving cleanup.
    Operates on ContentBlock.text only.
    """
    if not text:
        return ""

    # 1. Unicode normalization (fullwidth, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove invisible zero-width characters
    for ch in ZERO_WIDTH_CHARACTERS:
        text = text.replace(ch, "")

    # 3. Repair PDF hyphenation across line breaks
    #    "exam-\nple" → "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 4. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 5. Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # 6. Collapse excessive newlines (preserve paragraph boundaries)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()