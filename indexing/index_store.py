import pickle
from typing import Dict
from chunking.chunk_schema import Chunk


class IndexStore:
    """
    Stores mappings and metadata alongside FAISS index.
    """

    def __init__(self):
        self.faiss_id_to_embedding_id: Dict[int, str] = {}
        self.embedding_id_to_chunk: Dict[str, Chunk] = {}
        self.document_to_faiss_ids: Dict[str, set[int]] = {}

    def add(self, faiss_id: int, embedding_id: str, chunk: Chunk):
        self.faiss_id_to_embedding_id[faiss_id] = embedding_id
        self.embedding_id_to_chunk[embedding_id] = chunk
        self.document_to_faiss_ids.setdefault(
            chunk.document_id, set()
        ).add(faiss_id)

    def delete_document(self, document_id: str):
        faiss_ids = self.document_to_faiss_ids.pop(document_id, set())
        for fid in faiss_ids:
            eid = self.faiss_id_to_embedding_id.pop(fid, None)
            if eid:
                self.embedding_id_to_chunk.pop(eid, None)
        return faiss_ids

    def persist(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "IndexStore":
        with open(path, "rb") as f:
            return pickle.load(f)
