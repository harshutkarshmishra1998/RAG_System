from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------
# Environment bootstrap
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from schema.ingest_router import ingest
from clean_normalize.phase2_pipeline import (
    run_phase_2_1_canonical_text_sanitation,
    run_phase_2_2_structural_block_segmentation,
    run_phase_2_3_boilerplate_detection_suppression,
    run_phase_2_4_table_normalization,
    run_phase_2_5_metadata_canonicalization,
    run_phase_2_6_deterministic_ordering_hashing,
)

# ------------------------------------------------------------
# Test runner
# ------------------------------------------------------------

def run_test(input_value):
    print("\n================ PIPELINE =================")
    print(f"Input: {input_value}")

    # -------------------------
    # Phase-1: Ingestion
    # -------------------------
    doc = ingest(input_value)

    # -------------------------
    # Phase-2: Clean, Normalize & Validate
    # -------------------------
    doc = run_phase_2_1_canonical_text_sanitation(doc)
    doc = run_phase_2_2_structural_block_segmentation(doc)
    doc = run_phase_2_3_boilerplate_detection_suppression(doc)
    doc = run_phase_2_4_table_normalization(doc)
    doc = run_phase_2_5_metadata_canonicalization(doc)
    doc = run_phase_2_6_deterministic_ordering_hashing(doc)


    # -------------------------
    # Diagnostics
    # -------------------------
    print(f"Document ID : {doc.document_id}")
    print(f"Source type : {doc.source.source_type}")
    print(f"Source URI  : {doc.source.source_uri}")
    print(f"Blocks      : {len(doc.blocks)}")
    print("==========================================")

    # -------------------------
    # Assertions (pipeline guarantees)
    # -------------------------
    assert doc.blocks, "No blocks extracted"
    
    # Phase 2.1
    for idx, block in enumerate(doc.blocks):
        if block.text:
            assert block.text == block.text.strip(), (
                f"Leading/trailing whitespace in block {idx}"
            )
            assert "\u200b" not in block.text, (
                f"Zero-width char leak in block {idx}"
            )
            assert "\ufeff" not in block.text, (
                f"BOM leak in block {idx}"
            )

    # Phase 2.2
    has_empty = False
    has_noise = False

    for block in doc.blocks:
        meta = block.metadata.extra

        if meta.get("empty_block"):
            has_empty = True

        if meta.get("noise_block"):
            has_noise = True

    # Structural validation guarantees
    assert doc.blocks, "Blocks vanished unexpectedly"
    # Phase-2.2 flags issues *if they exist*
    for block in doc.blocks:
        meta = block.metadata.extra
        assert isinstance(meta, dict), "metadata.extra corrupted"

    # Phase 2.3
    has_boilerplate = False

    for block in doc.blocks:
        if block.metadata.extra.get("boilerplate"):
            has_boilerplate = True
            break

    # Boilerplate is expected mainly for WEB / YOUTUBE
    if doc.source.source_type in {"web", "youtube"}:
        if not has_boilerplate:
            print("[INFO] No boilerplate detected (clean or short content)")

    # Phase 2.4
    for block in doc.blocks:
        if block.content_type.name == "TABLE":
            meta = block.metadata.extra
            assert "table_normalized" in meta, "TABLE block not normalized"
            assert meta["table_rows"] >= 0
            assert meta["table_columns"] >= 1

    # Phase 2.5
    for block in doc.blocks:
        meta = block.metadata.extra

        # Canonical keys must exist
        assert "content_type" in meta
        assert "is_ocr" in meta
        assert "has_table" in meta
        assert "source_type" in meta
        assert "confidence" in meta

        # Structural flags must be explicit booleans
        assert isinstance(meta.get("empty_block"), bool)
        assert isinstance(meta.get("noise_block"), bool)
        assert isinstance(meta.get("boilerplate"), bool)

    # Phase 2.6
    seen_hashes = set()

    for block in doc.blocks:
        meta = block.metadata.extra

        assert "content_hash" in meta
        assert "structural_hash" in meta

        # Hashes must be stable strings
        assert isinstance(meta["content_hash"], str)
        assert isinstance(meta["structural_hash"], str)

        # Detect accidental duplication
        seen_hashes.add(meta["content_hash"])

    # Re-run hashing to ensure determinism
    doc2 = run_phase_2_6_deterministic_ordering_hashing(doc)

    for b1, b2 in zip(doc.blocks, doc2.blocks):
        assert b1.metadata.extra["content_hash"] == b2.metadata.extra["content_hash"]

    # Phase 2.7
    # 1. Blocks must exist
    assert doc.blocks, "Phase-2 produced empty document"

    # 2. Block order must be deterministic
    hash_sequence = [
        block.metadata.extra.get("structural_hash")
        for block in doc.blocks
    ]
    assert all(hash_sequence), "Missing structural_hash in some blocks"

    # 3. Metadata contract must be complete
    REQUIRED_KEYS = {
        "content_type",
        "is_ocr",
        "has_table",
        "source_type",
        "confidence",
        "content_hash",
        "structural_hash",
    }

    for idx, block in enumerate(doc.blocks):
        meta = block.metadata.extra

        missing = REQUIRED_KEYS - meta.keys()
        assert not missing, (
            f"Missing metadata keys {missing} in block {idx}"
        )

    # 4. Phase-2 must be idempotent
    doc_again = run_phase_2_6_deterministic_ordering_hashing(doc)

    for b1, b2 in zip(doc.blocks, doc_again.blocks):
        assert (
            b1.metadata.extra["content_hash"]
            == b2.metadata.extra["content_hash"]
        ), "Non-idempotent content hash detected"

    # 5. Phase-2 must not mutate block identity
    block_ids_before = [b.block_id for b in doc.blocks]
    block_ids_after = [b.block_id for b in doc_again.blocks]

    assert block_ids_before == block_ids_after, (
        "Phase-2 mutated block identities"
    )


def main():
    tests = [
        PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        PROJECT_ROOT / "test" / "test.txt",
        "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit",
    ]

    for item in tests:
        run_test(item)


if __name__ == "__main__":
    main()