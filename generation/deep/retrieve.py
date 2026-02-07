from typing import Dict, List
from retrieval.phase5_2_retrieval import run_phase_5_2_retrieval


def multi_query_retrieval_phase_7_4_2(state: Dict) -> Dict:
    if "index" not in state:
        return {
            **state,
            "multi_retrieval_results": [],
            "deep_error": "missing_index",
        }
    
    index = state["index"]
    chunks_by_embedding_id = state["chunks_by_embedding_id"]
    model_id = state["model_id"]
    k = state.get("k", 5)

    results: List[Dict] = []

    for q in state["expanded_queries"]:
        retrieval = run_phase_5_2_retrieval(
            query=q,
            index=index,
            chunks_by_embedding_id=chunks_by_embedding_id,
            model_id=model_id,
            k=k,
        )

        results.append({
            "query": q,
            "status": retrieval.status,
            "chunks": retrieval.results,
        })

    return {
        **state,
        "multi_retrieval_results": results,
    }