from typing import Dict

from context_assembly.assemble import assemble_context_phase_6_3
from context_assembly.token_budget import enforce_token_budget_phase_6_4
from context_assembly.package import package_context_phase_6_5

def deep_context_assembly_phase_7_4_5(state: Dict) -> Dict:
    normalized = {
        "query": state["query"],
        "confidence": state["confidence"],
        "status": "success",
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "embedding_id": c.embedding_id,
                "document_id": c.document_id,
                "text": c.text,
                "score": c.score,
                "length_chars": len(c.text),
                "length_tokens": len(c.text.split()),
            }
            for c in state["deep_reranked_chunks"]
        ],
    }

    assembled = assemble_context_phase_6_3(
        normalized=normalized,
        reranked={"reranked_chunks": normalized["chunks"]},
        max_tokens=600,
    )

    budgeted = enforce_token_budget_phase_6_4(
        assembled=assembled,
        max_tokens=500,
    )

    packaged = package_context_phase_6_5(budgeted=budgeted)

    return {
        **state,
        "deep_context_text": packaged["context_text"],
        "deep_context_tokens": packaged["context_tokens"],
    }
