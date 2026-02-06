# To run:
# pytest test/test_phase5_1_faiss_insertion_all_sources.py

from pathlib import Path
import tempfile
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

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


def _run_phase1_to_phase3(input_value):
    """Full pipeline (Phase-1 → Phase-3)"""
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


def test_phase_5_1_faiss_insertion_all_sources():
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
        print("\n================ PHASE-5.1 INDEX PIPELINE ================")
        print(f"Input: {input_value}")

        # -----------------------------
        # Phase-1 → Phase-3
        # -----------------------------
        doc, chunks = _run_phase1_to_phase3(input_value)
        assert chunks, "Chunks must not be empty"

        # -----------------------------
        # Phase-5.0 — real embeddings
        # -----------------------------
        cache = EmbeddingCache()
        embeddings = embed_chunks(chunks, model, cache)
        assert embeddings, "Embeddings must not be empty"

        # -----------------------------
        # Phase-5.1 — FAISS insertion
        # -----------------------------
        index = FaissIndex(dimension=model.dimension)

        run_phase_5_1_faiss_insertion(
            chunks=chunks,
            embeddings=embeddings,
            index=index,
        )

        index.persist("vector_store/faiss/index")

        # -----------------------------
        # Sanity search
        # -----------------------------
        sample_embedding_id = chunks[0].metadata["embedding_id"]
        query_vector = embeddings[sample_embedding_id]

        results = index.search(query_vector, k=5)
        assert results, "Inserted vectors must be searchable"

        # -----------------------------
        # Document-level hygiene
        # -----------------------------
        index.delete_by_document(doc.document_id)

        after_delete = index.search(
            query_vector,
            k=5,
            filters={"document_id": doc.document_id},
        )
        assert not after_delete, "Deleted document vectors must not appear"

        # -----------------------------
        # Persistence hygiene
        # -----------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "faiss_index"
            index.persist(str(path))

            reloaded = FaissIndex(dimension=model.dimension)
            reloaded.load(str(path))

            after_reload = reloaded.search(
                query_vector,
                k=5,
                filters={"document_id": doc.document_id},
            )
            assert not after_reload