import hashlib


def compute_embedding_id(chunk_hash: str, model_id: str) -> str:
    """
    Deterministic embedding identity.

    Same chunk + same model -> same embedding_id
    """
    raw = f"{chunk_hash}::{model_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()