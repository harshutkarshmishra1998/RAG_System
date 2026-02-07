from langgraph.graph import StateGraph, END

# -------- Control plane --------
# from asyncio import graph
from generation.gate import generation_gate_phase_7_0
from generation.scope import decompose_query_phase_7_1
from generation.strategy import select_strategy_phase_7_2
from generation.cheap import cheap_answer_generation_phase_7_3

# -------- Deep path --------
from generation.deep.expand import expand_query_phase_7_4_1
from generation.deep.retrieve import multi_query_retrieval_phase_7_4_2
from generation.deep.dedupe import dedupe_chunks_phase_7_4_3
from generation.deep.rerank import rerank_deep_chunks_phase_7_4_4
from generation.deep.assemble import deep_context_assembly_phase_7_4_5

from generation.deep.answer import deep_answer_generation_phase_7_4_6


def route_after_strategy(state: dict) -> str:
    strategy = state.get("strategy")

    if strategy == "deep":
        return "deep_expand"

    if strategy == "cheap":
        return "cheap_generate"

    return END



def build_phase7_graph():
    graph = StateGraph(dict)

    # ==============================
    # Core control plane
    # ==============================
    graph.add_node("gate", generation_gate_phase_7_0)
    graph.add_node("scope", decompose_query_phase_7_1)
    graph.add_node("strategy", select_strategy_phase_7_2)

    # ==============================
    # Cheap path
    # ==============================
    graph.add_node("cheap_generate", cheap_answer_generation_phase_7_3)

    # ==============================
    # Deep path
    # ==============================
    graph.add_node("deep_expand", expand_query_phase_7_4_1)
    graph.add_node("deep_retrieve", multi_query_retrieval_phase_7_4_2)
    graph.add_node("deep_dedupe", dedupe_chunks_phase_7_4_3)
    graph.add_node("deep_rerank", rerank_deep_chunks_phase_7_4_4)
    graph.add_node("deep_assemble", deep_context_assembly_phase_7_4_5)
    graph.add_node("deep_answer", deep_answer_generation_phase_7_4_6)

    # ==============================
    # Entry point
    # ==============================
    graph.set_entry_point("gate")

    # ==============================
    # Linear control flow
    # ==============================
    graph.add_edge("gate", "scope")
    graph.add_edge("scope", "strategy")

    # ==============================
    # Conditional branching
    # ==============================
    graph.add_conditional_edges(
        "strategy",
        route_after_strategy,
    )

    # ==============================
    # Cheap path termination
    # ==============================
    graph.add_edge("cheap_generate", END)

    # ==============================
    # Deep path chain
    # ==============================
    graph.add_edge("deep_expand", "deep_retrieve")
    graph.add_edge("deep_retrieve", "deep_dedupe")
    graph.add_edge("deep_dedupe", "deep_rerank")
    graph.add_edge("deep_rerank", "deep_assemble")
    graph.add_edge("deep_assemble", "deep_answer")
    graph.add_edge("deep_answer", END)

    return graph.compile()
