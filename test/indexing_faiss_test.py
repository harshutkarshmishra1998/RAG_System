# To run this file, use following command from project root:
# pytest test/indexing_faiss_test.py

from pathlib import Path
from dotenv import load_dotenv
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Phase-1
from schema.ingest_router import ingest

# Phase-2
from clean_normalize.phase2_pipeline import (
    run_phase_2_1_canonical_text_sanitation,
    run_phase_2_2_structural_block_segmentation,
    run_phase_2_3_boilerplate_detection_suppression,
    run_phase_2_4_table_normalization,
    run_phase_2_5_metadata_canonicalization,
    run_phase_2_6_deterministic_ordering_hashing,
)

# Phase-3
from chunking.phase3_pipeline import (
    run_phase_3_2_block_aware_chunking,
    run_phase_3_3_adaptive_chunk_assembly,
    run_phase_3_4_chunk_metadata_finalization,
)

# Phase-4.1
from embeddings.embedding_spec import EmbeddingModelSpec
from embeddings.embedding_cache import EmbeddingCache
from embeddings.embedding_pipeline import embed_chunks

# Phase-4.2 / 4.3
from indexing.faiss_index import FaissIndex


def run_pipeline(input_value):
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

    model = EmbeddingModelSpec(
        model_id="test-model",
        dimension=8,
        provider="test",
    )

    cache = EmbeddingCache()
    embeddings = embed_chunks(chunks, model, cache)

    return doc, chunks, embeddings, model


def test_indexing_all_sources_and_hygiene():
    tests = [
        PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        PROJECT_ROOT / "test" / "test.txt",
        "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit",
    ]

    for t in tests:
        print("\n================ INDEX PIPELINE =================")
        print(f"Input: {t}")

        doc, chunks, embeddings, model = run_pipeline(t)

        index = FaissIndex(dimension=model.dimension)
        index.add(chunks, embeddings)

        query_embedding = embeddings[chunks[0].metadata["embedding_id"]]

        # Basic search
        results = index.search(query_embedding, k=5)
        assert results

        # Filter by document
        filtered = index.search(
            query_embedding,
            k=5,
            filters={"document_id": doc.document_id},
        )
        assert filtered

        # Deletion hygiene
        index.delete_by_document(doc.document_id)

        after_delete = index.search(
            query_embedding,
            k=5,
            filters={"document_id": doc.document_id},
        )
        assert not after_delete

        # Persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "faiss_index")
            index.persist(path)

            reloaded = FaissIndex(dimension=model.dimension)
            reloaded.load(path)

            results_after_reload = reloaded.search(query_embedding, k=5)
            # deleted doc should still be gone
            assert not results_after_reload
