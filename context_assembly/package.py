from typing import Dict, List


def package_context_phase_6_5(
    *,
    budgeted: Dict,
) -> Dict:
    """
    Phase-6.5 — Context Packaging

    Produces final model-ready context text
    with explicit chunk boundaries.
    """

    chunks: List[Dict] = budgeted["final_chunks"]

    blocks: List[str] = []

    for i, c in enumerate(chunks, 1):
        block = (
            f"[CHUNK {i} | id={c['chunk_id']} | score={c['score']:.4f}]\n"
            f"{c['text']}"
        )
        blocks.append(block)

    context_text = "\n\n---\n\n".join(blocks)

    return {
        "context_text": context_text,
        "context_chunks": chunks,
        "context_tokens": budgeted["final_tokens"],
        "packaging_metadata": {
            "format": "chunk_delimited_v1",
            "num_chunks": len(chunks),
        },
    }