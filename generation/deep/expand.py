from typing import Dict, List

MAX_EXPANSIONS = 3
MAX_TOKENS_PER_QUERY = 15


def expand_query_phase_7_4_1(state: Dict) -> Dict:
    query = state["query"]

    expansions: List[str] = [query]

    ql = query.lower()

    if "why" in ql:
        expansions.append(query.replace("why", "what problem does"))

    if "how" not in ql:
        expansions.append(f"How does {query}")

    clean: List[str] = []
    for q in expansions:
        q = " ".join(q.split()[:MAX_TOKENS_PER_QUERY])
        if q not in clean:
            clean.append(q)

    return {
        **state,
        "expanded_queries": clean[:MAX_EXPANSIONS],
        "expansion_strategy": "heuristic",
    }