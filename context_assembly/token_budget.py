from typing import Dict, List


def enforce_token_budget_phase_6_4(
    *,
    assembled: Dict,
    max_tokens: int,
) -> Dict:
    """
    Phase-6.4 — Token Budget Enforcement (hard guard)

    Ensures context never exceeds token limit,
    even if upstream logic miscalculates.
    """

    chunks: List[Dict] = assembled["context_chunks"]

    final_chunks: List[Dict] = []
    trimmed: List[str] = []
    total_tokens = 0

    for c in chunks:
        tokens = c.get("length_tokens", 0)

        if tokens <= 0:
            trimmed.append(c["chunk_id"])
            continue

        if total_tokens + tokens > max_tokens:
            trimmed.append(c["chunk_id"])
            continue

        final_chunks.append(c)
        total_tokens += tokens

    return {
        "final_chunks": final_chunks,
        "final_tokens": total_tokens,
        "budget_metadata": {
            "max_tokens": max_tokens,
            "enforced": True,
            "trimmed": trimmed,
        },
    }
