# chunking/adaptive_chunk_assembly.py

import hashlib
from typing import List
from chunking.chunk_schema import Chunk


MIN_CHARS = 500
MAX_CHARS = 3500
OVERLAP_CHARS = 200


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def adaptive_chunk_assembly(chunks: List[Chunk]) -> List[Chunk]:
    """
    Phase-3.3: Adaptive Chunk Assembly

    - Merge micro-chunks
    - Split oversized chunks
    - Add deterministic overlap
    - Assign stable chunk_hash
    """

    # --------------------------------------------------
    # Step 1: Merge undersized chunks (same document only)
    # --------------------------------------------------
    merged: List[Chunk] = []
    buffer = None

    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue

        # Merge only if both are small and compatible
        if (
            len(buffer.text) < MIN_CHARS
            and len(chunk.text) < MIN_CHARS
            and buffer.document_id == chunk.document_id
            and set(buffer.content_types) == set(chunk.content_types)
        ):
            buffer.text += "\n\n" + chunk.text
            buffer.block_ids.extend(chunk.block_ids)
            buffer.page_end = chunk.page_end
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    # --------------------------------------------------
    # Step 2: Split oversized chunks with overlap
    # --------------------------------------------------
    final_chunks: List[Chunk] = []
    chunk_index = 0

    for chunk in merged:
        text = chunk.text

        if len(text) <= MAX_CHARS:
            chunk.chunk_index = chunk_index
            chunk.metadata["chunk_hash"] = _hash_text(text)
            final_chunks.append(chunk)
            chunk_index += 1
            continue

        start = 0
        while start < len(text):
            end = start + MAX_CHARS
            window = text[start:end]

            new_chunk = Chunk(
                document_id=chunk.document_id,
                chunk_index=chunk_index,
                text=window,
                block_ids=chunk.block_ids,
                content_types=chunk.content_types,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                metadata=dict(chunk.metadata),
            )

            new_chunk.metadata["chunk_hash"] = _hash_text(window)
            final_chunks.append(new_chunk)

            chunk_index += 1
            start = end - OVERLAP_CHARS

    return final_chunks