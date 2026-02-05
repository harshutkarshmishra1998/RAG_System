# chunking/chunk_metadata_finalization.py

from typing import List
from chunking.chunk_schema import Chunk


def finalize_chunk_metadata(chunks: List[Chunk]) -> List[Chunk]:
    """
    Phase-3.4: Chunk Metadata Finalization

    - Propagates confidence deterministically
    - Freezes lineage metadata
    - Enforces chunk-level invariants
    """

    for chunk in chunks:
        meta = chunk.metadata

        # --------------------------------------------------
        # Source lineage
        # --------------------------------------------------
        meta.setdefault("document_id", chunk.document_id)
        meta.setdefault("chunk_index", chunk.chunk_index)
        meta.setdefault("block_ids", list(chunk.block_ids))

        # --------------------------------------------------
        # Confidence propagation (conservative)
        # --------------------------------------------------
        # If already set, respect it
        if "confidence" not in meta:
            # Default neutral confidence
            meta["confidence"] = 1.0

        # --------------------------------------------------
        # Content summary flags
        # --------------------------------------------------
        meta.setdefault(
            "content_types",
            list(set(chunk.content_types)),
        )

        # --------------------------------------------------
        # Hard invariants (raise immediately if broken)
        # --------------------------------------------------
        assert chunk.text.strip(), "Invariant violation: empty chunk text"
        assert chunk.block_ids, "Invariant violation: chunk without block_ids"
        assert "chunk_hash" in meta, "Invariant violation: missing chunk_hash"

    return chunks
