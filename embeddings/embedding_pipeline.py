# from typing import List
# from chunking.chunk_schema import Chunk
# from embeddings.embedding_spec import EmbeddingModelSpec
# from embeddings.embedding_identity import compute_embedding_id
# from embeddings.embedding_cache import EmbeddingCache


# def _fake_embed(text: str, dimension: int) -> List[float]:
#     """
#     Deterministic fake embedding.
#     Used ONLY for Phase-4.1 tests.
#     """
#     base = sum(ord(c) for c in text) % 1000
#     return [(base + i) / 1000.0 for i in range(dimension)]


# def embed_chunks(
#     chunks: List[Chunk],
#     model: EmbeddingModelSpec,
#     cache: EmbeddingCache,
# ):
#     """
#     Phase-4.1 embedding pipeline.

#     Returns:
#         Dict[embedding_id, vector]
#     """

#     embeddings = {}

#     for chunk in chunks:
#         chunk_hash = chunk.metadata["chunk_hash"]
#         embedding_id = compute_embedding_id(chunk_hash, model.model_id)

#         # Cache lookup
#         cached = cache.get(embedding_id)
#         if cached is not None:
#             embeddings[embedding_id] = cached
#             continue

#         # Fake deterministic embedding
#         vector = _fake_embed(chunk.text, model.dimension)

#         cache.set(embedding_id, vector)
#         embeddings[embedding_id] = vector

#     return embeddings


from typing import List
from chunking.chunk_schema import Chunk
from embeddings.embedding_spec import EmbeddingModelSpec
from embeddings.embedding_identity import compute_embedding_id
from embeddings.embedding_cache import EmbeddingCache


def _fake_embed(text: str, dimension: int) -> List[float]:
    """
    Deterministic fake embedding.
    Used ONLY for Phase-4.1 tests.
    """
    base = sum(ord(c) for c in text) % 1000
    return [(base + i) / 1000.0 for i in range(dimension)]


def embed_chunks(
    chunks: List[Chunk],
    model: EmbeddingModelSpec,
    cache: EmbeddingCache,
):
    """
    Phase-4.1 embedding pipeline.

    Side effects (INTENTIONAL):
    - attaches embedding_id to chunk.metadata
    """

    embeddings = {}

    for chunk in chunks:
        # -----------------------------
        # 1. Compute embedding identity
        # -----------------------------
        chunk_hash = chunk.metadata["chunk_hash"]
        embedding_id = compute_embedding_id(
            chunk_hash=chunk_hash,
            model_id=model.model_id,
        )

        # 🔑 THIS IS THE LINE YOU ASKED ABOUT
        chunk.metadata["embedding_id"] = embedding_id

        # -----------------------------
        # 2. Cache lookup
        # -----------------------------
        cached = cache.get(embedding_id)
        if cached is not None:
            embeddings[embedding_id] = cached
            continue

        # -----------------------------
        # 3. Compute embedding (fake)
        # -----------------------------
        vector = _fake_embed(
            text=chunk.text,
            dimension=model.dimension,
        )

        # -----------------------------
        # 4. Cache write
        # -----------------------------
        cache.set(embedding_id, vector)
        embeddings[embedding_id] = vector

    return embeddings