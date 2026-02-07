from typing import Dict, List


def dedupe_chunks_phase_7_4_3(state: Dict) -> Dict:
    seen = set()
    deduped: List[Dict] = []

    for group in state["multi_retrieval_results"]:
        for chunk in group["chunks"]:
            cid = chunk.chunk_id
            if cid not in seen:
                seen.add(cid)
                deduped.append(chunk)

    return {
        **state,
        "deduped_chunks": deduped,
        # "deduped_count": len(deduped),
    }