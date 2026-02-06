from dataclasses import dataclass
from typing import List

from retrieval.phase5_2_retrieval import RetrievalResult, RetrievedChunk
from retrieval.phase5_3_calibration import RetrievalDiagnostics


@dataclass(frozen=True)
class RetrievalEnvelope:
    """
    Phase-5.4 → Phase-6.1 bridge.

    Combines:
    - retrieval result (chunks)
    - calibration diagnostics (confidence)
    """

    query: str
    status: str
    confidence: str
    diagnostics: RetrievalDiagnostics
    chunks: List[RetrievedChunk]


def build_retrieval_envelope(
    *,
    retrieval: RetrievalResult,
    diagnostics: RetrievalDiagnostics,
) -> RetrievalEnvelope:
    return RetrievalEnvelope(
        query=retrieval.query,
        status=retrieval.status,
        confidence=diagnostics.confidence,
        diagnostics=diagnostics,
        chunks=retrieval.results,
    )