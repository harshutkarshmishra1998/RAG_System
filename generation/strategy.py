# def select_strategy_phase_7_2(
#     *,
#     confidence: str,
#     scope_type: str,
#     num_chunks: int,
# ) -> dict:
#     """
#     Phase-7.2 — Cheap vs Deep strategy selection
#     """

#     # Hard rules first (no LLM guesswork)
#     if confidence == "low":
#         strategy = "deep"
#     elif scope_type == "multi":
#         strategy = "deep"
#     elif num_chunks < 2:
#         strategy = "deep"
#     else:
#         strategy = "cheap"

#     return {
#         "strategy": strategy
#     }

# generation/strategy.py

def select_strategy_phase_7_2(state: dict) -> dict:
    confidence = state["confidence"]
    scope_type = state["scope_type"]
    num_chunks = state["num_chunks"]

    if confidence == "low":
        strategy = "deep"
    elif scope_type == "multi":
        strategy = "deep"
    elif num_chunks < 2:
        strategy = "deep"
    else:
        strategy = "cheap"

    return {
        **state,
        "strategy": strategy,
    }