# generation/judge.py

from typing import Dict, List

# def judge_phase_7_5(state: Dict) -> Dict:
#     """
#     Phase-7.5 — Judge & Retry Logic

#     Evaluates the generated answer and decides:
#     - accept
#     - retry
#     - final_with_warning
#     """

#     query: str = state["query"]
#     strategy_used: str = state["strategy"]

#     retry_count: int = state.get("retry_count", 0)
#     max_retries: int = state.get("max_retries", 1)

#     # Pick answer
#     answer = None
#     if strategy_used == "cheap":
#         answer = state.get("cheap_answer")
#     elif strategy_used == "deep":
#         answer = state.get("deep_answer")

#     issues: List[str] = []

#     # -------------------------------
#     # Safety checks
#     # -------------------------------
#     if answer is None or not answer.get("text"):
#         issues.append("empty_answer")

#     if not answer.get("used_context", False):
#         issues.append("answer_not_grounded")

#     if len(answer.get("text", "")) < 80:
#         issues.append("answer_too_short")

#     # -------------------------------
#     # Coverage heuristic
#     # -------------------------------
#     sub_queries = state.get("sub_queries", [])
#     for sq in sub_queries:
#         if sq.lower() not in answer.get("text", "").lower():
#             issues.append(f"missing_coverage:{sq}")

#     # -------------------------------
#     # Decision logic
#     # -------------------------------
#     if not issues:
#         verdict = "accept"
#         confidence = "high"

#     else:
#         if strategy_used == "cheap" and retry_count < max_retries:
#             verdict = "retry"
#             confidence = "low"
#         else:
#             verdict = "final_with_warning"
#             confidence = "low"

#     # -------------------------------
#     # Write judge output
#     # -------------------------------
#     state.update(
#         {
#             "judge_verdict": verdict,
#             "judge_issues": issues,
#             "final_confidence": confidence,
#             "retry_count": retry_count + (1 if verdict == "retry" else 0),
#         }
#     )

#     return state

HARD_MAX_RETRIES = 2


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def judge_phase_7_5(state: dict) -> dict:
    """Evaluate generated answer quality and prevent unbounded retry loops."""
    verdict = "pass"
    issues = []

    strategy = state.get("strategy")
    retry_count = max(0, _safe_int(state.get("retry_count", 0), 0))
    requested_max = max(0, _safe_int(state.get("max_retries", 1), 1))
    max_retries = min(requested_max, HARD_MAX_RETRIES)

    if strategy == "cheap":
        answer = state.get("cheap_answer", {}).get("text", "")
    else:
        answer = state.get("deep_answer", {}).get("text", "")

    if not answer or len(answer.strip()) < 30:
        issues.append("answer_too_short")
        if retry_count < max_retries:
            verdict = "retry"
            retry_count += 1
        else:
            verdict = "final_with_warning"

    return {
        **state,
        "judge_verdict": verdict,
        "judge_issues": issues,
        "final_confidence": "high" if verdict == "pass" else "low",
        "retry_count": retry_count,
        "max_retries": max_retries,
    }
