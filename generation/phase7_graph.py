# from langgraph.graph import StateGraph, END

# from generation.gate import generation_gate_phase_7_0
# from generation.scope import decompose_query_phase_7_1
# from generation.strategy import select_strategy_phase_7_2


# def build_phase7_graph():
#     graph = StateGraph(dict)

#     graph.add_node("gate", generation_gate_phase_7_0)
#     graph.add_node("scope", lambda s: decompose_query_phase_7_1(s["query"]))
#     graph.add_node("strategy", lambda s: select_strategy_phase_7_2(
#         confidence=s["confidence"],
#         scope_type=s["scope_type"],
#         num_chunks=s["num_chunks"],
#     ))

#     graph.set_entry_point("gate")

#     graph.add_edge("gate", "scope")
#     graph.add_edge("scope", "strategy")
#     graph.add_edge("strategy", END)

#     return graph.compile()

# generation/phase7_graph.py

from langgraph.graph import StateGraph, END

from generation.gate import generation_gate_phase_7_0
from generation.scope import decompose_query_phase_7_1
from generation.strategy import select_strategy_phase_7_2
from generation.cheap import cheap_answer_generation_phase_7_3


def build_phase7_graph():
    graph = StateGraph(dict)

    graph.add_node("gate", generation_gate_phase_7_0)
    graph.add_node("scope", decompose_query_phase_7_1)
    graph.add_node("strategy", select_strategy_phase_7_2)
    graph.add_node("cheap_generate", cheap_answer_generation_phase_7_3)

    graph.set_entry_point("gate")

    graph.add_edge("gate", "scope")
    graph.add_edge("scope", "strategy")
    graph.add_edge("strategy", "cheap_generate")
    graph.add_edge("cheap_generate", END)

    return graph.compile()