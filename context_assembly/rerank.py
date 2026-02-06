from typing import Dict, List, Tuple


# --------------------------------------------------
# Heuristic signals
# --------------------------------------------------
BOILERPLATE_MARKERS = [
    "archived at",
    "retrieved from",
    "external links",
    "references",
    "doi:",
    "wayback machine",
]

DEFINITION_PATTERNS = [
    " is an ",
    " is a ",
    " refers to ",
    " defined as ",
]


def _contains_any(text: str, patterns: List[str]) -> bool:
    text_l = text.lower()
    return any(p in text_l for p in patterns)


def rerank_chunks_phase_6_2(
    normalized: Dict,
) -> Dict:
    """
    Phase-6.2 — Heuristic reranking
    """

    chunks = normalized["chunks"]
    query = normalized["query"].lower()

    scored: List[Tuple[float, Dict]] = []
    suppressed: List[str] = []

    for c in chunks:
        score = c["score"]
        text = c["text"].lower()

        boost = 0.0

        # --------------------------------------------------
        # Definition-first boost
        # --------------------------------------------------
        if any(p in text for p in DEFINITION_PATTERNS) and query in text:
            boost += 0.25

        # --------------------------------------------------
        # Early-position boost
        # --------------------------------------------------
        pos = c.get("position")
        if isinstance(pos, int) and pos <= 3:
            boost += 0.10

        # --------------------------------------------------
        # Boilerplate demotion
        # --------------------------------------------------
        if _contains_any(text, BOILERPLATE_MARKERS):
            boost -= 0.40

        final_rank_score = score + boost
        scored.append((final_rank_score, c))

    # --------------------------------------------------
    # Sort by adjusted score (descending)
    # --------------------------------------------------
    scored.sort(key=lambda x: x[0], reverse=True)

    reranked_chunks = [c for _, c in scored]

    return {
        "reranked_chunks": reranked_chunks,
        "rerank_metadata": {
            "strategy": "heuristic_v1",
            "suppressed": suppressed,
        },
    }
