from typing import Dict, List


def assemble_context_phase_6_3(
    *,
    normalized: Dict,
    reranked: Dict,
    max_tokens: int = 1200,
) -> Dict:
    """
    Phase-6.3 — Context Window Assembly

    Strategy:
    - Greedy, rank-ordered inclusion
    - Enforce token budget
    - No text mutation
    """

    reranked_chunks: List[Dict] = reranked["reranked_chunks"]
    confidence: str = normalized["confidence"]

    # -----------------------------
    # Confidence gate
    # -----------------------------
    if confidence == "low":
        return {
            "context_chunks": [],
            "context_tokens": 0,
            "context_mode": "degraded",
            "assembly_metadata": {
                "max_tokens": max_tokens,
                "strategy": "greedy_by_rank_v1",
                "dropped": [c["chunk_id"] for c in reranked_chunks],
            },
        }

    selected: List[Dict] = []
    dropped: List[str] = []
    total_tokens = 0

    for c in reranked_chunks:
        tokens = c.get("length_tokens", 0)

        if tokens <= 0:
            dropped.append(c["chunk_id"])
            continue

        if total_tokens + tokens > max_tokens:
            dropped.append(c["chunk_id"])
            continue

        selected.append(c)
        total_tokens += tokens

    context_mode = "normal" if selected else "empty"

    return {
        "context_chunks": selected,
        "context_tokens": total_tokens,
        "context_mode": context_mode,
        "assembly_metadata": {
            "max_tokens": max_tokens,
            "strategy": "greedy_by_rank_v1",
            "dropped": dropped,
        },
    }
