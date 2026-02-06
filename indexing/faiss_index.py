# #faiss_index.py

# import json
# import os
# from pathlib import Path

# import faiss
# import numpy as np
# from typing import List, Dict, Any

# from indexing.index_base import IndexBackend
# from indexing.index_store import IndexStore
# from chunking.chunk_schema import Chunk


# class FaissIndex(IndexBackend):
#     """
#     FAISS-backed vector index with metadata filtering and hygiene.
#     """

#     def __init__(self, dimension: int):
#         self.dimension = dimension
#         self.index = faiss.IndexFlatL2(dimension)
#         self.store = IndexStore()

#     # --------------------------------------------------
#     # Add
#     # --------------------------------------------------
#     def add(self, chunks: List[Chunk], embeddings: Dict[str, List[float]]) -> None:
#         for chunk in chunks:
#             embedding_id = chunk.metadata["embedding_id"]

#             # Idempotent add
#             if embedding_id in self.store.embedding_id_to_chunk:
#                 continue

#             vector = np.array(
#                 embeddings[embedding_id], dtype="float32"
#             ).reshape(1, -1)

#             faiss_id = self.index.ntotal
#             self.index.add(vector) #type: ignore
#             self.store.add(faiss_id, embedding_id, chunk)

#         # Hygiene invariant
#         assert self.index.ntotal == len(self.store.faiss_id_to_embedding_id)

#     # --------------------------------------------------
#     # Delete
#     # --------------------------------------------------
#     def delete_by_document(self, document_id: str) -> None:
#         self.store.delete_document(document_id)

#     # --------------------------------------------------
#     # Search
#     # --------------------------------------------------
#     def search(
#         self,
#         query_embedding,
#         k: int,
#         filters: Dict[str, Any] | None = None,
#     ) -> List[Chunk]:

#         query = np.array(
#             query_embedding, dtype="float32"
#         ).reshape(1, -1)

#         distances, indices = self.index.search(query, k * 3) #type: ignore

#         results: List[Chunk] = []

#         for fid in indices[0]:
#             if fid == -1:
#                 continue

#             embedding_id = self.store.faiss_id_to_embedding_id.get(fid)
#             if not embedding_id:
#                 continue

#             chunk = self.store.embedding_id_to_chunk.get(embedding_id)
#             if not chunk:
#                 continue  # deleted

#             if filters and not self._passes_filters(chunk, filters):
#                 continue

#             results.append(chunk)
#             if len(results) >= k:
#                 break

#         return results

#     # --------------------------------------------------
#     # Persistence
#     # --------------------------------------------------
#     def persist(self, path: str) -> None:
#         faiss.write_index(self.index, path + ".faiss")
#         self.store.persist(path + ".meta")

#     def load(self, path: str) -> None:
#         self.index = faiss.read_index(path + ".faiss")
#         self.store = IndexStore.load(path + ".meta")

#     # --------------------------------------------------
#     # Filtering
#     # --------------------------------------------------
#     def _passes_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
#         meta = chunk.metadata

#         for key, value in filters.items():

#             # Exact match
#             if key in ("source_type", "document_id"):
#                 if meta.get(key) != value:
#                     return False

#             # Content type intersection
#             elif key == "content_types":
#                 if not set(meta.get("content_types", [])).intersection(set(value)):
#                     return False

#             # Confidence threshold
#             elif key == "confidence_gte":
#                 if meta.get("confidence", 0.0) < value:
#                     return False

#             # Page range overlap
#             elif key == "page_range":
#                 start, end = value
#                 page_start = meta.get("page_start")
#                 page_end = meta.get("page_end")

#                 if page_start is None or page_end is None:
#                     return False

#                 if page_end < start or page_start > end:
#                     return False

#             else:
#                 raise ValueError(f"Unsupported filter key: {key}")

#         return True
    
#     # --------------------------------------------------
#     # Phase-5.4 — Observability helpers
#     # --------------------------------------------------
#     def size(self) -> int:
#         return self.index.ntotal

#     def document_counts(self) -> Dict[str, int]:
#         counts: Dict[str, int] = {}
#         for chunk in self.store.embedding_id_to_chunk.values():
#             doc_id = chunk.metadata.get("document_id")
#             counts[doc_id] = counts.get(doc_id, 0) + 1 #type: ignore
#         return counts


#     def sample_chunks(self, n: int = 5) -> List[Chunk]:
#         return list(self.store.embedding_id_to_chunk.values())[:n]

# indexing/faiss_index.py

import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

from indexing.index_base import IndexBackend
from indexing.index_store import IndexStore
from chunking.chunk_schema import Chunk


class FaissIndex(IndexBackend):
    """
    FAISS-backed vector index with metadata filtering and hygiene.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.store = IndexStore()

    # --------------------------------------------------
    # Add
    # --------------------------------------------------
    def add(self, chunks: List[Chunk], embeddings: Dict[str, List[float]]) -> None:
        for chunk in chunks:
            embedding_id = chunk.metadata["embedding_id"]

            # Idempotent add
            if embedding_id in self.store.embedding_id_to_chunk:
                continue

            vector = np.array(
                embeddings[embedding_id], dtype="float32"
            ).reshape(1, -1)

            faiss_id = self.index.ntotal
            self.index.add(vector)  # type: ignore
            self.store.add(faiss_id, embedding_id, chunk)

        # Hygiene invariant
        assert self.index.ntotal == len(self.store.faiss_id_to_embedding_id)

    # --------------------------------------------------
    # Search (FIXED — RETURNS SCORES)
    # --------------------------------------------------
    def search(
        self,
        query_embedding: List[float],
        k: int,
        filters: Dict[str, Any] | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Returns: List[(Chunk, similarity_score)]
        similarity_score ∈ (0, 1], higher = better
        """

        if self.index.ntotal == 0:
            return []

        query = np.array(
            query_embedding, dtype="float32"
        ).reshape(1, -1)

        # Search wider, then filter
        distances, indices = self.index.search(query, k * 3)  # type: ignore

        results: List[Tuple[Chunk, float]] = []

        for dist, fid in zip(distances[0], indices[0]):
            if fid == -1:
                continue

            embedding_id = self.store.faiss_id_to_embedding_id.get(fid)
            if not embedding_id:
                continue

            chunk = self.store.embedding_id_to_chunk.get(embedding_id)
            if not chunk:
                continue

            if filters and not self._passes_filters(chunk, filters):
                continue

            # Convert L2 distance → similarity
            similarity = 1.0 / (1.0 + float(dist))

            results.append((chunk, similarity))

            if len(results) >= k:
                break

        return results

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
    def persist(self, path: str) -> None:
        faiss.write_index(self.index, path + ".faiss")
        self.store.persist(path + ".meta")

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path + ".faiss")
        self.store = IndexStore.load(path + ".meta")

    # --------------------------------------------------
    # Filtering
    # --------------------------------------------------
    def _passes_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        meta = chunk.metadata

        for key, value in filters.items():
            if key in ("source_type", "document_id"):
                if meta.get(key) != value:
                    return False

            elif key == "content_types":
                if not set(meta.get("content_types", [])).intersection(set(value)):
                    return False

            elif key == "confidence_gte":
                if meta.get("confidence", 0.0) < value:
                    return False

            elif key == "page_range":
                start, end = value
                ps, pe = meta.get("page_start"), meta.get("page_end")
                if ps is None or pe is None:
                    return False
                if pe < start or ps > end:
                    return False

            else:
                raise ValueError(f"Unsupported filter key: {key}")

        return True

    # --------------------------------------------------
    # Observability
    # --------------------------------------------------
    def size(self) -> int:
        return self.index.ntotal
    
    # --------------------------------------------------
    # Delete (required by IndexBackend)
    # --------------------------------------------------
    def delete_by_document(self, document_id: str) -> None:
        """
        Phase-4/5 hygiene hook.
        Logical delete handled by IndexStore.
        """
        self.store.delete_document(document_id)
