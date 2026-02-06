# from typing import List, Dict, Literal, Optional
# from dataclasses import dataclass


# RetrievalStatus = Literal["success", "low_confidence", "empty"]


# @dataclass(frozen=True)
# class RetrievedChunk:
#     chunk_id: str
#     embedding_id: str
#     document_id: str
#     score: float
#     text: str
#     metadata: Dict


# @dataclass(frozen=True)
# class RetrievalResult:
#     status: RetrievalStatus
#     query: str
#     results: List[RetrievedChunk]


# from typing import Optional
# from embeddings.open_source_embedder import OpenSourceEmbedder
# from indexing.faiss_index import FaissIndex
# from chunking.chunk_schema import Chunk
# import re


# def _normalize_query(query: str) -> str:
#     """
#     Phase-5.2.0 — deterministic query normalization.
#     """
#     query = query.strip().lower()
#     query = re.sub(r"\s+", " ", query)
#     query = re.sub(r"[?]+$", "?", query)
#     return query


# def run_phase_5_2_retrieval(
#     *,
#     query: str,
#     index: FaissIndex,
#     chunks_by_embedding_id: Dict[str, Chunk],
#     model_id: str,
#     k: int = 5,
#     min_similarity: float = 0.25,
#     filters: Optional[Dict] = None,
# ) -> RetrievalResult:
#     """
#     Phase-5.2 — Query embedding + FAISS retrieval.

#     STRICTLY follows the retrieval contract.
#     """

#     # -----------------------------
#     # 5.2.0 — Normalize query
#     # -----------------------------
#     normalized_query = _normalize_query(query)

#     if not normalized_query:
#         return RetrievalResult(
#             status="empty",
#             query=query,
#             results=[],
#         )

#     # -----------------------------
#     # 5.2.1 — Embed query
#     # -----------------------------
#     embedder = OpenSourceEmbedder(model_id)
#     query_vector = embedder.embed(normalized_query)

#     # -----------------------------
#     # 5.2.2 — FAISS search
#     # -----------------------------
#     raw_results = index.search(
#         query_vector,
#         k=k,
#         filters=filters,
#     )

#     if not raw_results:
#         return RetrievalResult(
#             status="empty",
#             query=query,
#             results=[],
#         )

#     # -----------------------------
#     # 5.2.3 — Similarity gating
#     # -----------------------------
#     scores = [
#         c.metadata.get("score", 0.0)
#         for c in raw_results
#     ]

#     if not scores:
#         return RetrievalResult(
#             status="empty",
#             query=query,
#             results=[],
#         )

#     max_score = max(scores)

#     if max_score < min_similarity:
#         return RetrievalResult(
#             status="low_confidence",
#             query=query,
#             results=[],
#         )

#     # -----------------------------
#     # 5.2.4 — Result assembly
#     # -----------------------------
#     results: List[RetrievedChunk] = []

#     for chunk in raw_results:
#         embedding_id = chunk.metadata.get("embedding_id")
#         if not embedding_id:
#             continue

#         results.append(
#             RetrievedChunk(
#                 chunk_id=chunk.chunk_id,
#                 embedding_id=embedding_id,
#                 document_id=chunk.metadata.get("document_id"), #type: ignore
#                 score=chunk.metadata.get("score", 0.0),
#                 text=chunk.text,
#                 metadata=chunk.metadata,
#             )
#         )

#     if not results:
#         return RetrievalResult(
#             status="empty",
#             query=query,
#             results=[],
#         )

#     return RetrievalResult(
#         status="success",
#         query=query,
#         results=results,
#     )

# retrieval/phase5_2_retrieval.py

from typing import List, Dict, Literal, Optional, Tuple
from dataclasses import dataclass
import re

from embeddings.open_source_embedder import OpenSourceEmbedder
from indexing.faiss_index import FaissIndex
from chunking.chunk_schema import Chunk


RetrievalStatus = Literal["success", "low_confidence", "empty"]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    embedding_id: str
    document_id: str
    score: float
    text: str
    metadata: Dict


@dataclass(frozen=True)
class RetrievalResult:
    status: RetrievalStatus
    query: str
    results: List[RetrievedChunk]


def _normalize_query(query: str) -> str:
    query = query.strip().lower()
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"[?]+$", "?", query)
    return query


def run_phase_5_2_retrieval(
    *,
    query: str,
    index: FaissIndex,
    chunks_by_embedding_id: Dict[str, Chunk],  # kept for compatibility
    model_id: str,
    k: int = 5,
    min_similarity: float = 0.25,
    filters: Optional[Dict] = None,
) -> RetrievalResult:

    normalized_query = _normalize_query(query)
    if not normalized_query:
        return RetrievalResult("empty", query, [])

    embedder = OpenSourceEmbedder(model_id)
    query_vector = embedder.embed(normalized_query)

    raw_results: List[Tuple[Chunk, float]] = index.search(
        query_vector,
        k=k,
        filters=filters,
    )

    if not raw_results:
        return RetrievalResult("empty", query, [])

    scores = [score for _, score in raw_results]
    max_score = max(scores)

    if max_score < min_similarity:
        return RetrievalResult("low_confidence", query, [])

    results: List[RetrievedChunk] = []

    for chunk, score in raw_results:
        embedding_id = chunk.metadata.get("embedding_id")
        if not embedding_id:
            continue

        results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                embedding_id=embedding_id,
                document_id=chunk.metadata["document_id"],  # type: ignore
                score=score,
                text=chunk.text,
                metadata=chunk.metadata,
            )
        )

    if not results:
        return RetrievalResult("empty", query, [])

    return RetrievalResult("success", query, results)


