from pydantic import BaseModel, ConfigDict


class EmbeddingModelSpec(BaseModel):
    """
    Logical embedding model descriptor.
    Provider-agnostic and deterministic.
    """

    model_id: str
    dimension: int
    provider: str

    # Fix pydantic v2 protected namespace warning
    model_config = ConfigDict(
        protected_namespaces=()
    )