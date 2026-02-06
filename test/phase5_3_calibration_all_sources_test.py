from operator import index
from pathlib import Path
from dotenv import load_dotenv

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


def test_phase_5_3_calibration_all_sources():
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

    for input_value in tests:
        print("\n================ PHASE-5.3 CALIBRATION =================")
        print(f"Input: {input_value}")

        # Phase-1 → Phase-3
        doc, chunks = _run_phase1_to_phase3(input_value)
        assert chunks

        # Phase-5.0
        cache = EmbeddingCache()
        embeddings = embed_chunks(chunks, model, cache)

        # Phase-5.1
        index = FaissIndex(dimension=model.dimension)
        run_phase_5_1_faiss_insertion(
            chunks=chunks,
            embeddings=embeddings,
            index=index,
        )

        # Iteration 1 → PDF
        # - build FAISS with PDF chunks (say 120)
        # - persist → index.faiss (120 vectors)

        # Iteration 2 → DOCX
        # - NEW FaissIndex()
        # - build FAISS with DOCX chunks only (say 80)
        # - persist → index.faiss (OVERWRITES previous)
        
        # Iteration 3 → TXT
        # - NEW FaissIndex()
        # - persist → index.faiss (overwrites again)

        # ...

        # Iteration 6 → Google Doc
        # - NEW FaissIndex()
        # - chunks are small → only 4 chunks
        # - persist → index.faiss (FINAL STATE)

        # # Phase-5.4 — persist vector store
        # index.persist(str(VECTOR_STORE_PATH))

        # # Phase-5.4 — load persisted vector store
        # loaded_index = FaissIndex(dimension=model.dimension)
        # loaded_index.load(str(VECTOR_STORE_PATH))

        chunks_by_embedding_id = {
            c.metadata["embedding_id"]: c for c in chunks
        }

        # Phase-5.2
        retrieval = run_phase_5_2_retrieval(
            query="gradient descent",
            # index=loaded_index,
            index=index,
            chunks_by_embedding_id=chunks_by_embedding_id,
            model_id=model.model_id,
            k=5,
        )

        # Phase-5.3
        diagnostics = run_phase_5_3_calibration(retrieval)

        # -----------------------------
        # Contract assertions
        # -----------------------------
        assert diagnostics.confidence in {
            "high",
            "medium",
            "low",
            "empty",
        }

        assert diagnostics.phase5_2_status == retrieval.status
        assert diagnostics.num_results >= 0
        assert diagnostics.top_score >= 0.0
        assert diagnostics.mean_score >= 0.0
        assert diagnostics.score_spread >= 0.0
