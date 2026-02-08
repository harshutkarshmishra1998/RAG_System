from langgraph.graph import StateGraph, END

from generation.gate import generation_gate_phase_7_0
from generation.scope import decompose_query_phase_7_1
from generation.strategy import select_strategy_phase_7_2
from generation.cheap import cheap_answer_generation_phase_7_3

from generation.deep.expand import expand_query_phase_7_4_1
from generation.deep.retrieve import multi_query_retrieval_phase_7_4_2
from generation.deep.dedupe import dedupe_chunks_phase_7_4_3
from generation.deep.rerank import rerank_deep_chunks_phase_7_4_4
from generation.deep.assemble import deep_context_assembly_phase_7_4_5
from generation.deep.answer import deep_answer_generation_phase_7_4_6

from generation.judge import judge_phase_7_5


def route_after_strategy(state: dict) -> str:
    if state["strategy"] == "cheap":
        return "cheap_generate"
    if state["strategy"] == "deep":
        return "deep_expand"
    return END


# def route_after_judge(state: dict) -> str:
#     # Retry ONLY triggers deep path
#     if state.get("judge_verdict") == "retry":
#         return "deep_expand"
#     return END

def route_after_judge(state: dict) -> str:
    verdict = state.get("judge_verdict")
    confidence = state.get("final_confidence")

    # Retry on explicit retry
    if verdict == "retry" or confidence == "low":
        state["strategy"] = "deep"
        return "deep_expand"

    return END

def finalize_phase7_state(state: dict) -> dict:
    return {
        "generation_mode": state.get("generation_mode", "abstain"),
        "scope_type": state.get("scope_type", "single"),
        "sub_queries": state.get("sub_queries", []),
        "strategy": state.get("strategy", "cheap"),
        **state,
    }

def build_phase7_graph():
    graph = StateGraph(dict)

    # Control plane
    graph.add_node("gate", generation_gate_phase_7_0)
    graph.add_node("scope", decompose_query_phase_7_1)
    graph.add_node("strategy", select_strategy_phase_7_2)

    # Cheap
    graph.add_node("cheap_generate", cheap_answer_generation_phase_7_3)

    # Deep
    graph.add_node("deep_expand", expand_query_phase_7_4_1)
    graph.add_node("deep_retrieve", multi_query_retrieval_phase_7_4_2)
    graph.add_node("deep_dedupe", dedupe_chunks_phase_7_4_3)
    graph.add_node("deep_rerank", rerank_deep_chunks_phase_7_4_4)
    graph.add_node("deep_assemble", deep_context_assembly_phase_7_4_5)
    graph.add_node("deep_generate", deep_answer_generation_phase_7_4_6)

    # Judge
    graph.add_node("judge", judge_phase_7_5)

    graph.set_entry_point("gate")

    # Control plane flow
    graph.add_edge("gate", "scope")
    graph.add_edge("scope", "strategy")

    graph.add_conditional_edges(
        "strategy",
        lambda s: "cheap_generate" if s["strategy"] == "cheap" else "deep_expand",
    )

    # Cheap path
    graph.add_edge("cheap_generate", "judge")

    # Deep path
    graph.add_edge("deep_expand", "deep_retrieve")
    graph.add_edge("deep_retrieve", "deep_dedupe")
    graph.add_edge("deep_dedupe", "deep_rerank")
    graph.add_edge("deep_rerank", "deep_assemble")
    graph.add_edge("deep_assemble", "deep_generate")
    graph.add_edge("deep_generate", "judge")

    # Judge routing (ONLY ONE)
    graph.add_conditional_edges("judge", route_after_judge)

    return graph.compile()

