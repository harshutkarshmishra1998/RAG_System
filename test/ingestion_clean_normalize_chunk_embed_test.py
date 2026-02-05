from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# -------------------------
# Phase-1: Ingestion
# -------------------------
from schema.ingest_router import ingest

# -------------------------
# Phase-2: Cleaning & Normalization
# -------------------------
from clean_normalize.phase2_pipeline import (
    run_phase_2_1_canonical_text_sanitation,
    run_phase_2_2_structural_block_segmentation,
    run_phase_2_3_boilerplate_detection_suppression,
    run_phase_2_4_table_normalization,
    run_phase_2_5_metadata_canonicalization,
    run_phase_2_6_deterministic_ordering_hashing,
)

# -------------------------
# Phase-3: Chunking
# -------------------------
from chunking.phase3_pipeline import (
    run_phase_3_2_block_aware_chunking,
    run_phase_3_3_adaptive_chunk_assembly,
    run_phase_3_4_chunk_metadata_finalization,
)

# -------------------------
# Phase-4.1: Embeddings
# -------------------------
from embeddings.embedding_spec import EmbeddingModelSpec
from embeddings.embedding_cache import EmbeddingCache
from embeddings.embedding_pipeline import embed_chunks
from embeddings.embedding_identity import compute_embedding_id


def run_test(input_value):
    print("\n================ PIPELINE =================")
    print(f"Input: {input_value}")

    # -------------------------
    # Phase-1
    # -------------------------
    doc = ingest(input_value)

    # -------------------------
    # Phase-2
    # -------------------------
    doc = run_phase_2_1_canonical_text_sanitation(doc)
    doc = run_phase_2_2_structural_block_segmentation(doc)
    doc = run_phase_2_3_boilerplate_detection_suppression(doc)
    doc = run_phase_2_4_table_normalization(doc)
    doc = run_phase_2_5_metadata_canonicalization(doc)
    doc = run_phase_2_6_deterministic_ordering_hashing(doc)

    # -------------------------
    # Phase-3
    # -------------------------
    chunks = run_phase_3_2_block_aware_chunking(doc)
    chunks = run_phase_3_3_adaptive_chunk_assembly(chunks)
    chunks = run_phase_3_4_chunk_metadata_finalization(chunks)

    assert chunks, "No chunks produced"

    # -------------------------
    # Phase-4.1
    # -------------------------
    model = EmbeddingModelSpec(
        model_id="test-embedding-model",
        dimension=8,
        provider="test",
    )

    cache = EmbeddingCache()

    embeddings = embed_chunks(
        chunks=chunks,
        model=model,
        cache=cache,
    )

    # -------------------------
    # Assertions (CORRECT)
    # -------------------------
    assert embeddings, "No embeddings produced"

    for chunk in chunks:
        chunk_hash = chunk.metadata["chunk_hash"]

        embedding_id = compute_embedding_id(
            chunk_hash=chunk_hash,
            model_id=model.model_id,
        )

        assert embedding_id in embeddings, (
            f"Missing embedding for chunk_hash={chunk_hash}"
        )

        vector = embeddings[embedding_id]
        assert len(vector) == model.dimension


def main():
    tests = [
        PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        PROJECT_ROOT / "test" / "test.txt",
        "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit",
    ]

    for t in tests:
        run_test(t)


if __name__ == "__main__":
    main()