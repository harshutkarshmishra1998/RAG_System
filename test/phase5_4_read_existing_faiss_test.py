from pathlib import Path

from indexing.faiss_index import FaissIndex


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This must already exist from a previous ingestion run
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "faiss" / "index"


def test_read_existing_faiss_index():
    """
    Phase-5.4:
    This test ONLY verifies that an already-created FAISS index
    can be loaded and read.

    It does NOT:
    - ingest documents
    - embed chunks
    - modify FAISS
    """

    # Sanity check: index files must exist
    assert (VECTOR_STORE_PATH.with_suffix(".faiss")).exists(), \
        "FAISS index file not found. Run ingestion first."

    assert (VECTOR_STORE_PATH.with_suffix(".meta")).exists(), \
        "FAISS metadata file not found. Run ingestion first."

    # Load existing FAISS index
    index = FaissIndex(dimension=384)
    index.load(str(VECTOR_STORE_PATH))

    # Basic read assertions
    assert index.index.ntotal > 0, "FAISS index is empty"

    assert len(index.store.embedding_id_to_chunk) == index.index.ntotal

    # Optional: print info for visibility (pytest will show with -s)
    print("\n================ EXISTING FAISS INDEX ================")
    print(f"Total vectors : {index.index.ntotal}")

    doc_ids = set(
        chunk.metadata.get("document_id")
        for chunk in index.store.embedding_id_to_chunk.values()
    )

    print(f"Documents     : {len(doc_ids)}")
    for doc_id in doc_ids:
        print(f" - {doc_id}")
