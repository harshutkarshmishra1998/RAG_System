from typing import List
from sentence_transformers import SentenceTransformer


class OpenSourceEmbedder:
    """
    Open-source embedding model.
    Shared by tests and production.
    Loaded once and cached by HuggingFace.
    """

    _MODEL_CACHE = {}

    def __init__(self, model_name: str):
        if model_name not in self._MODEL_CACHE:
            self._MODEL_CACHE[model_name] = SentenceTransformer(
                model_name,
                device="cpu",
            )
        self.model = self._MODEL_CACHE[model_name]
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_id = model_name

    def embed(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()