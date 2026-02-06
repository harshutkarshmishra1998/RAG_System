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


def run_phase1_to_phase3(input_value):
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


def build_faiss_index_all_sources():
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

    # 🔒 ONE shared FAISS index
    index = FaissIndex(dimension=model.dimension)

    for input_value in tests:
        print(f"\n================ INGESTING =================")
        print(f"Source: {input_value}")

        _, chunks = run_phase1_to_phase3(input_value)
        print(f"Chunks created: {len(chunks)}")

        cache = EmbeddingCache()
        embeddings = embed_chunks(chunks, model, cache)

        run_phase_5_1_faiss_insertion(
            chunks=chunks,
            embeddings=embeddings,
            index=index,
        )

        print(f"Total vectors so far: {index.index.ntotal}")

    # 🔒 Persist ONCE (no overwrite loop)
    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.persist(str(VECTOR_STORE_PATH))

    print("\n================ DONE =================")
    print(f"FAISS vectors stored: {index.index.ntotal}")
    print(f"Saved to: {VECTOR_STORE_PATH}.faiss")


if __name__ == "__main__":
    build_faiss_index_all_sources()
