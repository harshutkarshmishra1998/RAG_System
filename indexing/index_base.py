from abc import ABC, abstractmethod
from typing import List, Dict, Any
from chunking.chunk_schema import Chunk


class IndexBackend(ABC):
    """
    Phase-4 Index contract.
    All index implementations MUST obey this.
    """

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the index.
        Must be idempotent based on chunk_hash.
        """
        pass

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        """
        Remove all chunks belonging to a document.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding,
        k: int,
        filters: Dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """
        Return top-k chunks matching the query.
        Must return Chunk objects with metadata intact.
        """
        pass

    @abstractmethod
    def persist(self, path: str) -> None:
        """
        Persist index to disk.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load index from disk.
        """
        pass