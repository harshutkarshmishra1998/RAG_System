from typing import List, Dict
from chunking.chunk_schema import Chunk
from embeddings.embedding_spec import EmbeddingModelSpec
from embeddings.embedding_identity import compute_embedding_id
from embeddings.embedding_cache import EmbeddingCache
from embeddings.open_source_embedder import OpenSourceEmbedder


def embed_chunks(
    chunks: List[Chunk],
    model: EmbeddingModelSpec,
    cache: EmbeddingCache,
) -> Dict[str, List[float]]:
    """
    Unified embedding pipeline.
    Tests and production use the SAME embedder.
    """

    embeddings: Dict[str, List[float]] = {}

    embedder = OpenSourceEmbedder(model.model_id)

    for chunk in chunks:
        chunk_hash = chunk.metadata["chunk_hash"]
        embedding_id = compute_embedding_id(
            chunk_hash=chunk_hash,
            model_id=model.model_id,
        )
        chunk.metadata["embedding_id"] = embedding_id

        cached = cache.get(embedding_id)
        if cached is not None:
            embeddings[embedding_id] = cached
            continue

        vector = embedder.embed(chunk.text)

        cache.set(embedding_id, vector)
        embeddings[embedding_id] = vector

    return embeddings