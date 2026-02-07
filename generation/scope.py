# import re

# MAX_SUB_QUERIES = 5

# def decompose_query_phase_7_1(query: str) -> dict:
#     """
#     Phase-7.1 — Scope detection & decomposition
#     """

#     # Very conservative split (no LLM yet)
#     parts = re.split(r"\band\b|\bor\b|,|\n", query)
#     parts = [p.strip() for p in parts if p.strip()]

#     if len(parts) > MAX_SUB_QUERIES:
#         parts = parts[:MAX_SUB_QUERIES]

#     return {
#         "original_query": query,
#         "sub_queries": parts,
#         "scope_type": "multi" if len(parts) > 1 else "single",
#     }

# generation/scope.py

import re

MAX_SUB_QUERIES = 5


def decompose_query_phase_7_1(state: dict) -> dict:
    query = state["query"]

    parts = re.split(r"\band\b|\bor\b|,|\n", query)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > MAX_SUB_QUERIES:
        parts = parts[:MAX_SUB_QUERIES]

    return {
        **state,
        "sub_queries": parts,
        "scope_type": "multi" if len(parts) > 1 else "single",
    }