from pathlib import Path
from dotenv import load_dotenv

# --------------------------------------------------
# Project setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "faiss" / "index"

# --------------------------------------------------
# Phase-1
# --------------------------------------------------
from schema.ingest_router import ingest

# --------------------------------------------------
# Phase-2
# --------------------------------------------------
from clean_normalize.phase2_pipeline import (
    run_phase_2_1_canonical_text_sanitation,
    run_phase_2_2_structural_block_segmentation,
    run_phase_2_3_boilerplate_detection_suppression,
    run_phase_2_4_table_normalization,
    run_phase_2_5_metadata_canonicalization,
    run_phase_2_6_deterministic_ordering_hashing,
)

# --------------------------------------------------
# Phase-3
# --------------------------------------------------
from chunking.phase3_pipeline import (
    run_phase_3_2_block_aware_chunking,
    run_phase_3_3_adaptive_chunk_assembly,
    run_phase_3_4_chunk_metadata_finalization,
)

# --------------------------------------------------
# Phase-5.0
# --------------------------------------------------
from embeddings.embedding_spec import EmbeddingModelSpec
from embeddings.embedding_cache import EmbeddingCache
from embeddings.embedding_pipeline import embed_chunks

# --------------------------------------------------
# Phase-5.1
# --------------------------------------------------
from indexing.faiss_index import FaissIndex
from indexing.phase5_1_faiss_pipeline import run_phase_5_1_faiss_insertion

# --------------------------------------------------
# Phase-5.2
# --------------------------------------------------
from retrieval.phase5_2_retrieval import run_phase_5_2_retrieval

# --------------------------------------------------
# Phase-5.3
# --------------------------------------------------
from retrieval.phase5_3_calibration import run_phase_5_3_calibration


def _run_phase1_to_phase3(input_value):
    doc = ingest(input_value)

    doc = run_phase_2_1_canonical_text_sanitation(doc)
    doc = run_phase_2_2_structural_block_segmentation(doc)
    doc = run_phase_2_3_boilerplate_detection_suppression(doc)
    doc = run_phase_2_4_table_normalization(doc)
    doc = run_phase_2_5_metadata_canonicalization(doc)
    doc = run_phase_2_6_deterministic_ordering_hashing(doc)

    chunks = run_phase_3_2_block_aware_chunking(doc)
    chunks = run_phase_3_3_adaptive_chunk_assembly(chunks)
    chunks = run_phase_3_4_chunk_metadata_finalization(chunks)

    return doc, chunks


def test_phase1_to_phase5_4_all_sources_e2e():
    tests = [
        PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        PROJECT_ROOT / "test" / "test.txt",
        "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit",
    ]

    model = EmbeddingModelSpec(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        provider="local",
    )

    # --------------------------------------------------
    # Phase-5.1 — ONE shared FAISS index
    # --------------------------------------------------
    index = FaissIndex(dimension=model.dimension)

    all_chunks = []

    for input_value in tests:
        print("\n================ INGEST =================")
        print(f"Source: {input_value}")

        _, chunks = _run_phase1_to_phase3(input_value)
        assert chunks, "Chunks must not be empty"

        cache = EmbeddingCache()
        embeddings = embed_chunks(chunks, model, cache)

        run_phase_5_1_faiss_insertion(
            chunks=chunks,
            embeddings=embeddings,
            index=index,
        )

        all_chunks.extend(chunks)

        print(f"Chunks added      : {len(chunks)}")
        print(f"Total vectors now: {index.index.ntotal}")

    # --------------------------------------------------
    # Phase-5.4 — Persist FAISS ONCE
    # --------------------------------------------------
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.persist(str(VECTOR_STORE_PATH))

    assert VECTOR_STORE_PATH.with_suffix(".faiss").exists()
    assert VECTOR_STORE_PATH.with_suffix(".meta").exists()

    # --------------------------------------------------
    # Phase-5.4 — Load FAISS into a NEW object
    # --------------------------------------------------
    loaded_index = FaissIndex(dimension=model.dimension)
    loaded_index.load(str(VECTOR_STORE_PATH))

    assert loaded_index.index.ntotal == index.index.ntotal

    chunks_by_embedding_id = {
        c.metadata["embedding_id"]: c for c in all_chunks
    }

    # --------------------------------------------------
    # Phase-5.2 — Retrieval (from persisted index)
    # --------------------------------------------------
    retrieval = run_phase_5_2_retrieval(
        query="gradient descent",
        index=loaded_index,
        chunks_by_embedding_id=chunks_by_embedding_id,
        model_id=model.model_id,
        k=5,
    )

    # --------------------------------------------------
    # Phase-5.3 — Calibration
    # --------------------------------------------------
    diagnostics = run_phase_5_3_calibration(retrieval)

    # --------------------------------------------------
    # Contract assertions
    # --------------------------------------------------
    assert diagnostics.confidence in {"high", "medium", "low", "empty"}
    assert diagnostics.phase5_2_status == retrieval.status
    assert diagnostics.num_results >= 0
    assert diagnostics.top_score >= 0.0
    assert diagnostics.mean_score >= 0.0
    assert diagnostics.score_spread >= 0.0

    print("\n================ FINAL =================")
    print(f"Total documents : {len(set(c.metadata['document_id'] for c in all_chunks))}")
    print(f"Total chunks    : {len(all_chunks)}")
    print(f"Total vectors   : {loaded_index.index.ntotal}")
