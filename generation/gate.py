# def generation_gate_phase_7_0(*, context_mode: str, confidence: str) -> dict:
#     """
#     Phase-7.0 — Generation Gate
#     """

#     if context_mode == "empty":
#         mode = "abstain"
#     elif confidence == "low":
#         mode = "cautious"
#     else:
#         mode = "normal"

#     return {
#         "generation_mode": mode
#     }

# generation/gate.py

# generation/gate.py

def generation_gate_phase_7_0(state: dict) -> dict:
    confidence = state.get("confidence", "low")
    context_mode = state.get("context_mode", "empty")

    if confidence == "high" and context_mode == "normal":
        mode = "normal"
    elif confidence in {"medium", "low"}:
        mode = "cautious"
    else:
        mode = "abstain"

    # 🔒 HARD CONTRACT INITIALIZATION
    return {
        **state,                      # ⬅️ preserve upstream
        "generation_mode": mode,
        "scope_type": None,
        "sub_queries": [],
        "strategy": None,
    }
