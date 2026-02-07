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
    context_mode = state["context_mode"]
    confidence = state["confidence"]

    if context_mode == "empty":
        mode = "abstain"
    elif confidence == "low":
        mode = "cautious"
    else:
        mode = "normal"

    # ✅ PRESERVE STATE
    return {
        **state,
        "generation_mode": mode,
    }
