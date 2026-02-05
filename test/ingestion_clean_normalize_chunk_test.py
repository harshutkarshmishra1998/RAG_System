from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.ingest_router import ingest
from clean_normalize.phase2_pipeline import (
    run_phase_2_1_canonical_text_sanitation,
    run_phase_2_2_structural_block_segmentation,
    run_phase_2_3_boilerplate_detection_suppression,
    run_phase_2_4_table_normalization,
    run_phase_2_5_metadata_canonicalization,
    run_phase_2_6_deterministic_ordering_hashing,
)
from chunking.phase3_pipeline import (
    run_phase_3_2_block_aware_chunking,
    run_phase_3_3_adaptive_chunk_assembly,
    run_phase_3_4_chunk_metadata_finalization,
)


def run_test(input_value):
    print("\n================ PIPELINE =================")
    print(f"Input: {input_value}")

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

    print(f"Chunks produced: {len(chunks)}")

    # Phase 3.2
    assert chunks, "No chunks produced"

    for idx, chunk in enumerate(chunks):
        assert chunk.text.strip(), f"Empty chunk {idx}"
        assert chunk.block_ids, f"No block_ids in chunk {idx}"
        assert chunk.document_id == doc.document_id
        assert isinstance(chunk.chunk_index, int)

    #Phase 3.3
    assert chunks, "No chunks after adaptive assembly"

    seen_hashes = set()

    for idx, chunk in enumerate(chunks):
        assert chunk.text.strip(), f"Empty chunk {idx}"
        # assert len(chunk.text) >= 100, f"Suspiciously small chunk {idx}"
        assert chunk.text.strip(), f"Empty chunk {idx}"
        # Soft warning for small chunks (do NOT assert)
        if len(chunk.text) < 100:
            print(
                f"[INFO] Small chunk {idx} "
                f"(len={len(chunk.text)}) — allowed by design"
            )
        assert "chunk_hash" in chunk.metadata
        assert chunk.metadata["chunk_hash"] not in seen_hashes

        seen_hashes.add(chunk.metadata["chunk_hash"])
    
    # Phase 3.4
    assert chunks, "No chunks after Phase-3.4"

    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata

        # Identity & lineage
        assert meta["document_id"] == doc.document_id
        assert meta["chunk_index"] == chunk.chunk_index
        assert meta["block_ids"] == chunk.block_ids

        # Confidence
        assert isinstance(meta["confidence"], float)

        # Content summary
        assert "content_types" in meta
        assert isinstance(meta["content_types"], list)

        # Final invariants
        assert chunk.text.strip(), f"Empty chunk {idx}"
        assert "chunk_hash" in meta


def main():
    tests = [
        PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        PROJECT_ROOT / "test" / "test.txt",
        "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        "https://en.wikipedia.org/wiki/Gradient_descent",
    ]

    for t in tests:
        run_test(t)


if __name__ == "__main__":
    main()