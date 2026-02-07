from typing import Dict


def rerank_deep_chunks_phase_7_4_4(state: Dict) -> Dict:
    chunks = state["deduped_chunks"]

    reranked = sorted(
        chunks,
        key=lambda c: c.score,
        reverse=True,
    )

    return {
        **state,
        "deep_reranked_chunks": reranked,
    }