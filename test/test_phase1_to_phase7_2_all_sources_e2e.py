from pathlib import Path
from dotenv import load_dotenv
import os

# --------------------------------------------------
# Project setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "faiss" / "index"

def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing environment variable: {key}")
    return value

ENV_KEYS = {
    "GROQ_API_KEY": "GROQ_API",
    "LANGCHAIN_API_KEY": "LANGCHAIN_API",
    "LANGCHAIN_TRACING_V2": "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT": "LANGCHAIN_PROJECT",
}

# ---- LOAD + EXPORT ----
_loaded = {}

for env_name, source_key in ENV_KEYS.items():
    value = require_env(source_key)
    os.environ[env_name] = value
    _loaded[env_name] = value

# ---- OPTIONAL: SAFE DEBUG (NO LEAKS) ----
if __name__ == "__main__":
    for k in _loaded:
        print(f"{k}: loaded")

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

# --------------------------------------------------
# Phase-5.4 → Phase-6.1 Bridge
# --------------------------------------------------
from context_assembly.bridge import build_retrieval_envelope
from context_assembly.normalize import normalize_envelope

# --------------------------------------------------
# Phase-6.2
# --------------------------------------------------
from context_assembly.rerank import rerank_chunks_phase_6_2

# --------------------------------------------------
# Phase-6.3
# --------------------------------------------------
from context_assembly.assemble import assemble_context_phase_6_3

# --------------------------------------------------
# Phase-6.4 & Phase-6.5
# --------------------------------------------------
from context_assembly.token_budget import enforce_token_budget_phase_6_4
from context_assembly.package import package_context_phase_6_5

# --------------------------------------------------
# Phase-7.0 → Phase-7.2
# --------------------------------------------------
from generation.phase7_graph import build_phase7_graph

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
        # PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        # PROJECT_ROOT / "test" / "test.txt",
        # "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        # "https://en.wikipedia.org/wiki/Gradient_descent",
        # "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit",
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
        query="MCP",
        index=loaded_index,
        chunks_by_embedding_id=chunks_by_embedding_id,
        model_id=model.model_id,
        k=5,
    )

    # --------------------------------------------------
    # DEBUG — Raw retrieval visibility
    # --------------------------------------------------
    print("\n================ PHASE 5.2 DEBUG =================")
    print(f"Retrieval status       : {retrieval.status}")
    print(f"Raw chunks returned    : {len(retrieval.results)}")

    if retrieval.results:
        print("Scores:")
        for r in retrieval.results:
            print(f"  chunk_id={r.chunk_id} score={r.score:.4f}")


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

    # --------------------------------------------------
    # Phase-5.4 → Phase-6.1 — Bridge
    # --------------------------------------------------
    envelope = build_retrieval_envelope(
        retrieval=retrieval,
        diagnostics=diagnostics,
    )

    assert envelope.confidence == diagnostics.confidence
    assert envelope.status == retrieval.status
    assert len(envelope.chunks) == diagnostics.num_results

    # --------------------------------------------------
    # Phase-6.1 — Normalization
    # --------------------------------------------------
    normalized = normalize_envelope(envelope)

    # --------------------------------------------------
    # Phase-6.1 assertions
    # --------------------------------------------------
    assert normalized["confidence"] == diagnostics.confidence
    assert normalized["status"] == retrieval.status
    assert len(normalized["chunks"]) == diagnostics.num_results

    for raw, norm in zip(envelope.chunks, normalized["chunks"]):
        # Identity
        assert norm["chunk_id"] == raw.chunk_id
        assert norm["embedding_id"] == raw.embedding_id
        assert norm["document_id"] == raw.document_id

        # TEXT MUST NOT CHANGE
        assert norm["text"] == raw.text

        # Score preserved
        assert norm["score"] == raw.score

        # Derived fields
        assert norm["length_chars"] == len(raw.text)
        assert norm["length_tokens"] > 0
    
    print("\n================ PHASE 6.1 =================")
    print(f"Query       : {normalized['query']}")
    print(f"Confidence  : {normalized['confidence']}")
    print(f"Top score   : {diagnostics.top_score}")
    print(f"Mean score  : {diagnostics.mean_score}")
    print(f"Spread      : {diagnostics.score_spread}")
    print(f"Num chunks  : {len(normalized['chunks'])}")

    for i, c in enumerate(normalized["chunks"], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Chunk ID : {c['chunk_id']}")
        print(f"Score    : {c['score']}")
        print(f"Text     : {c['text'][:300]}...")

    # --------------------------------------------------
    # Phase-6.2 — Reranking
    # --------------------------------------------------
    reranked = rerank_chunks_phase_6_2(normalized)

    reranked_chunks = reranked["reranked_chunks"]

    # --------------------------------------------------
    # Phase-6.2 assertions
    # --------------------------------------------------
    assert len(reranked_chunks) == len(normalized["chunks"])

    # Identity preserved
    assert {
        c["chunk_id"] for c in reranked_chunks
    } == {
        c["chunk_id"] for c in normalized["chunks"]
    }

    # Scores must not change
    for before, after in zip(
        sorted(normalized["chunks"], key=lambda x: x["chunk_id"]),
        sorted(reranked_chunks, key=lambda x: x["chunk_id"]),
    ):
        assert before["score"] == after["score"]

    print("\n================ PHASE 6.2 =================")
    for i, c in enumerate(reranked_chunks, 1):
        print(f"\n--- Reranked Chunk {i} ---")
        print(f"Chunk ID : {c['chunk_id']}")
        print(f"Score    : {c['score']}")
        print(f"Text     : {c['text'][:300]}...")

    # --------------------------------------------------
    # Phase-6.3 — Context Assembly
    # --------------------------------------------------
    assembled = assemble_context_phase_6_3(
        normalized=normalized,
        reranked=reranked,
        max_tokens=600,   # intentionally small to test trimming
    )

    context_chunks = assembled["context_chunks"]

    # ----------------------------
    # Phase-6.3 assertions
    # ----------------------------
    assert assembled["context_mode"] in {"normal", "degraded", "empty"}
    assert assembled["context_tokens"] <= 600

    # Identity preserved: context ⊆ reranked
    reranked_ids = {c["chunk_id"] for c in reranked["reranked_chunks"]}
    context_ids = {c["chunk_id"] for c in context_chunks}
    assert context_ids.issubset(reranked_ids)

    # No text mutation
    for c in context_chunks:
        assert isinstance(c["text"], str) and len(c["text"]) > 0

    # Token accounting is consistent
    assert assembled["context_tokens"] == sum(
        c["length_tokens"] for c in context_chunks
    )

    print("\n================ PHASE 6.3 =================")
    print(f"Context mode   : {assembled['context_mode']}")
    print(f"Token budget   : 600")
    print(f"Used tokens    : {assembled['context_tokens']}")
    print(f"Chunks kept    : {len(context_chunks)}")
    print(f"Chunks dropped : {len(assembled['assembly_metadata']['dropped'])}")

    for i, c in enumerate(context_chunks, 1):
        print(f"\n--- Context Chunk {i} ---")
        print(f"Chunk ID : {c['chunk_id']}")
        print(f"Score    : {c['score']}")
        print(f"Tokens   : {c['length_tokens']}")
        print(f"Text     : {c['text'][:300]}...")

    # --------------------------------------------------
    # Phase-6.4 — Token Budget Enforcement
    # --------------------------------------------------
    budgeted = enforce_token_budget_phase_6_4(
        assembled=assembled,
        max_tokens=500,
    )

    assert budgeted["final_tokens"] <= 500
    assert all(
        c["length_tokens"] > 0 for c in budgeted["final_chunks"]
    )

    print("\n================ PHASE 6.4 =================")
    print(f"Max tokens    : 500")
    print(f"Used tokens   : {budgeted['final_tokens']}")
    print(f"Chunks kept   : {len(budgeted['final_chunks'])}")
    print(f"Chunks trimmed: {len(budgeted['budget_metadata']['trimmed'])}")

    # --------------------------------------------------
    # Phase-6.5 — Context Packaging
    # --------------------------------------------------
    packaged = package_context_phase_6_5(
        budgeted=budgeted,
    )

    # ----------------------------
    # Phase-6.5 assertions
    # ----------------------------
    assert isinstance(packaged["context_text"], str)
    assert packaged["context_tokens"] == budgeted["final_tokens"]
    assert packaged["packaging_metadata"]["num_chunks"] == len(
        budgeted["final_chunks"]
    )

    for c in budgeted["final_chunks"]:
        assert c["text"] in packaged["context_text"]

    print("\n================ PHASE 6.5 =================")
    print(f"Context tokens : {packaged['context_tokens']}")
    print(f"Chunks packed  : {packaged['packaging_metadata']['num_chunks']}")
    print("\n--- CONTEXT PREVIEW ---")
    print(packaged["context_text"][:1000])

    # --------------------------------------------------
    # Phase-7.0 → Phase-7.2 — Generation Control Plane
    # --------------------------------------------------
    phase7 = build_phase7_graph()

    phase7_input = {
        # user input
        "query": "Explain MCP and why it exists",

        # phase-6 signals
        "confidence": normalized["confidence"],
        "context_mode": assembled["context_mode"],
        "num_chunks": len(packaged["context_chunks"]),
        "context_text": packaged["context_text"],
        "context_tokens": packaged["context_tokens"],

        # 🔑 REQUIRED FOR DEEP PATH
        "index": loaded_index,
        "chunks_by_embedding_id": chunks_by_embedding_id,
        "model_id": model.model_id,
    }


    phase7_output = phase7.invoke(phase7_input)

    # ----------------------------
    # Phase-7.0 → 7.2 invariants
    # ----------------------------
    assert phase7_output["generation_mode"] in {"normal", "cautious", "abstain"}
    assert phase7_output["scope_type"] in {"single", "multi"}
    assert isinstance(phase7_output["sub_queries"], list)
    assert len(phase7_output["sub_queries"]) >= 1
    assert phase7_output["strategy"] in {"cheap", "deep"}

    print("\n================ PHASE 7.0 → 7.2 =================")
    print(f"Generation mode : {phase7_output['generation_mode']}")
    print(f"Scope type      : {phase7_output['scope_type']}")
    print(f"Sub-queries     : {phase7_output['sub_queries']}")
    print(f"Strategy        : {phase7_output['strategy']}")

    # ==================================================
    # Phase-7.3 / Phase-7.4 — Agentic path validation
    # ==================================================

    if phase7_output["strategy"] == "cheap":
        # --------------------------------------------------
        # Phase-7.3 — Cheap Answer Generation
        # --------------------------------------------------
        assert "cheap_answer" in phase7_output
        cheap = phase7_output["cheap_answer"]

        assert cheap is not None
        assert isinstance(cheap["text"], str)
        assert len(cheap["text"]) > 0
        assert cheap["used_context"] is True
        assert isinstance(cheap["model"], str)

        # Deep path must NOT exist
        assert phase7_output.get("expanded_queries") is None
        assert phase7_output.get("multi_retrieval_results") is None
        assert phase7_output.get("deduped_chunks") is None
        assert phase7_output.get("deep_reranked_chunks") is None
        assert phase7_output.get("deep_context_text") is None

        print("\n================ PHASE 7.3 =================")
        print(f"Model used : {cheap['model']}")
        print("\n--- CHEAP ANSWER ---")
        print(cheap["text"])

    elif phase7_output["strategy"] == "deep":
        assert phase7_output.get("cheap_answer") is None
        # --------------------------------------------------
        # Phase-7.4 — Deep Path (FULL)
        # --------------------------------------------------

        # -------- Scope --------
        assert phase7_output["scope_type"] == "multi"
        assert len(phase7_output["sub_queries"]) > 1

        # -------- Expansion --------
        assert "expanded_queries" in phase7_output
        assert isinstance(phase7_output["expanded_queries"], list)
        assert len(phase7_output["expanded_queries"]) >= len(
            phase7_output["sub_queries"]
        )

        # -------- Retrieval --------
        assert "multi_retrieval_results" in phase7_output
        assert isinstance(phase7_output["multi_retrieval_results"], list)
        assert len(phase7_output["multi_retrieval_results"]) > 0

        retrieved_chunks = []
        for group in phase7_output["multi_retrieval_results"]:
            retrieved_chunks.extend(group["chunks"])

        assert len(retrieved_chunks) > 0

        # -------- Deduplication --------
        assert "deduped_chunks" in phase7_output
        assert isinstance(phase7_output["deduped_chunks"], list)

        deduped_ids = {c.chunk_id for c in phase7_output["deduped_chunks"]}
        assert len(deduped_ids) == len(phase7_output["deduped_chunks"])

        # -------- Reranking --------
        assert "deep_reranked_chunks" in phase7_output
        reranked = phase7_output["deep_reranked_chunks"]

        assert isinstance(reranked, list)
        assert len(reranked) > 0

        scores = [c.score for c in reranked]
        assert scores == sorted(scores, reverse=True)

        # -------- Context assembly (Phase-6 reuse) --------
        assert "deep_context_text" in phase7_output
        assert isinstance(phase7_output["deep_context_text"], str)
        assert len(phase7_output["deep_context_text"]) > 0

        assert "deep_context_tokens" in phase7_output
        assert isinstance(phase7_output["deep_context_tokens"], int)
        assert phase7_output["deep_context_tokens"] > 0

        # Cheap path must NOT execute
        assert phase7_output.get("cheap_answer") is None

        print("\n================ PHASE 7.4 =================")
        print("Deep path executed (agent decision)")
        print(f"Sub-queries       : {len(phase7_output['sub_queries'])}")
        print(f"Expanded queries  : {len(phase7_output['expanded_queries'])}")
        print(f"Retrieved chunks  : {len(retrieved_chunks)}")
        print(f"Deduped chunks    : {len(phase7_output['deduped_chunks'])}")
        print(f"Final chunks      : {len(reranked)}")
        print(f"Context tokens    : {phase7_output['deep_context_tokens']}")

        if phase7_output.get("deep_answer"):
            print("\n--- DEEP ANSWER ---")
            print(f"Model used : {phase7_output['deep_answer']['model']}")
            print(phase7_output["deep_answer"]["text"])
        
    else:
        raise AssertionError("Unknown strategy selected by Phase-7")





