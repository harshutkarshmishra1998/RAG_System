# generation/judge.py

from typing import Dict

MAX_JUDGE_RETRIES = 1

def judge_phase_7_5(state: Dict) -> Dict:
    retries = state.get("judge_retries", 0)

    # Select answer
    if state["strategy"] == "cheap":
        answer = state.get("cheap_answer", {}).get("text", "")
    else:
        answer = state.get("deep_answer", {}).get("text", "")

    verdict = "pass"
    issues = []

    if not answer or len(answer.strip()) < 5: 
        issues.append("answer_too_short")
        if retries < MAX_JUDGE_RETRIES:
            verdict = "retry"
            retries += 1
        else:
            verdict = "final_with_warning"

    # 🚨 HARD STOP
    if verdict == "retry" and retries >= MAX_JUDGE_RETRIES:
        verdict = "pass"
        issues.append("Max retries reached")

    return {
        **state,
        "strategy": next_strategy,
        "judge_verdict": verdict,
        "judge_issues": issues,
        "judge_retries": retries+1,
        # "final_confidence": "high" if verdict == "pass" else "low",
        "final_confidence": "low",
    }
