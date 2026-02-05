from uuid import uuid4
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Chunk(BaseModel):
    """
    Phase-3 Chunk object.
    This is the smallest retrievable unit.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int

    text: str
    block_ids: List[str]

    content_types: List[str]
    page_start: Optional[int]
    page_end: Optional[int]

    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)