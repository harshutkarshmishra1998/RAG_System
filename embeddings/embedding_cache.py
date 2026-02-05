from typing import Dict, List


class EmbeddingCache:
    """
    Write-through embedding cache.
    Phase-4.1 uses in-memory cache only.
    """

    def __init__(self):
        self._store: Dict[str, List[float]] = {}

    def get(self, embedding_id: str):
        return self._store.get(embedding_id)

    def set(self, embedding_id: str, vector: List[float]) -> None:
        self._store[embedding_id] = vector

    def __contains__(self, embedding_id: str) -> bool:
        return embedding_id in self._store