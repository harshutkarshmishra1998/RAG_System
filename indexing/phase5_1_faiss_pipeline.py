from typing import Dict, List
from chunking.chunk_schema import Chunk
from indexing.faiss_index import FaissIndex


def run_phase_5_1_faiss_insertion(
    *,
    chunks: List[Chunk],
    embeddings: Dict[str, List[float]],
    index: FaissIndex,
) -> FaissIndex:
    """
    Phase-5.1: Document embedding insertion into FAISS.

    Responsibilities:
    - Insert chunk vectors into FAISS
    - Enforce document-level hygiene
    - Do NOT embed
    - Do NOT search
    - Do NOT persist (caller decides)

    Assumptions (guaranteed by earlier phases):
    - Each chunk has `embedding_id` in metadata
    - embeddings dict is keyed by embedding_id
    """

    # -----------------------------
    # 1. Sanity checks (fail fast)
    # -----------------------------
    if not chunks:
        return index

    for chunk in chunks:
        if "embedding_id" not in chunk.metadata:
            raise ValueError(
                "Chunk missing embedding_id. "
                "Phase-5.0 must run before Phase-5.1."
            )

        embedding_id = chunk.metadata["embedding_id"]
        if embedding_id not in embeddings:
            raise ValueError(
                f"Embedding missing for embedding_id={embedding_id}"
            )

    # -----------------------------
    # 2. Insert into FAISS
    # -----------------------------
    index.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    return index