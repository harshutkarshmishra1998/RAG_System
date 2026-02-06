from typing import List
from context_assembly.bridge import RetrievalEnvelope


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def normalize_envelope(envelope: RetrievalEnvelope) -> dict:
    """
    Phase-6.1 — Normalize retrieved chunks
    while preserving calibration.
    """

    normalized_chunks = []

    for chunk in envelope.chunks:
        meta = chunk.metadata or {}

        normalized_chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "embedding_id": chunk.embedding_id,
                "score": chunk.score,
                "text": chunk.text,  # MUST remain unchanged
                "source": meta.get("source"),
                "section": meta.get("section"),
                "position": meta.get("position"),
                "length_chars": len(chunk.text),
                "length_tokens": _estimate_tokens(chunk.text),
            }
        )

    return {
        "query": envelope.query,
        "status": envelope.status,
        "confidence": envelope.confidence,
        "diagnostics": envelope.diagnostics,
        "chunks": normalized_chunks,
    }
