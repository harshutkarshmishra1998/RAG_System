from typing import Dict, List, Literal
from dataclasses import dataclass

from retrieval.phase5_2_retrieval import RetrievalResult


ConfidenceLevel = Literal["high", "medium", "low", "empty"]


@dataclass(frozen=True)
class RetrievalDiagnostics:
    query: str
    confidence: ConfidenceLevel
    top_score: float
    mean_score: float
    score_spread: float
    num_results: int
    phase5_2_status: str


# -----------------------------
# Thresholds (LOCKED)
# -----------------------------
MIN_ABSOLUTE = 0.30
HIGH_ABSOLUTE = 0.60
MIN_SPREAD = 0.10
DOMINANCE_MARGIN = 0.05


def run_phase_5_3_calibration(
    retrieval: RetrievalResult,
) -> RetrievalDiagnostics:
    """
    Phase-5.3 — Retrieval calibration & diagnostics.

    STRICTLY follows the Phase-5.3 design spec.
    """

    # -----------------------------
    # Status override gate
    # -----------------------------
    if retrieval.status == "empty":
        return RetrievalDiagnostics(
            query=retrieval.query,
            confidence="empty",
            top_score=0.0,
            mean_score=0.0,
            score_spread=0.0,
            num_results=0,
            phase5_2_status=retrieval.status,
        )

    if retrieval.status == "low_confidence":
        return RetrievalDiagnostics(
            query=retrieval.query,
            confidence="low",
            top_score=0.0,
            mean_score=0.0,
            score_spread=0.0,
            num_results=len(retrieval.results),
            phase5_2_status=retrieval.status,
        )

    # -----------------------------
    # Metric computation
    # -----------------------------
    scores: List[float] = [r.score for r in retrieval.results]

    if not scores:
        return RetrievalDiagnostics(
            query=retrieval.query,
            confidence="empty",
            top_score=0.0,
            mean_score=0.0,
            score_spread=0.0,
            num_results=0,
            phase5_2_status=retrieval.status,
        )

    top_score = max(scores)
    mean_score = sum(scores) / len(scores)
    score_spread = top_score - min(scores)
    num_results = len(scores)

    # -----------------------------
    # Confidence classification
    # -----------------------------
    # HIGH
    if (
        top_score >= HIGH_ABSOLUTE
        and score_spread >= MIN_SPREAD
        and top_score >= mean_score + DOMINANCE_MARGIN
        and num_results >= 1
    ):
        confidence: ConfidenceLevel = "high"

    # MEDIUM
    elif (
        top_score >= MIN_ABSOLUTE
        and (
            score_spread >= MIN_SPREAD
            or top_score >= mean_score
        )
        and num_results >= 1
    ):
        confidence = "medium"

    # LOW
    else:
        confidence = "low"

    return RetrievalDiagnostics(
        query=retrieval.query,
        confidence=confidence,
        top_score=top_score,
        mean_score=mean_score,
        score_spread=score_spread,
        num_results=num_results,
        phase5_2_status=retrieval.status,
    )
