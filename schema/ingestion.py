from __future__ import annotations

from enum import Enum
from uuid import uuid4
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# Enums
# ============================================================

class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    YOUTUBE = "youtube"
    GOOGLE_DRIVE = "google_drive"


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_OCR = "image_ocr"


# ============================================================
# Metadata models
# ============================================================

class PageMetadata(BaseModel):
    """
    Page-level metadata.
    Be consistent about page numbering (recommend: 1-based).
    """
    page_number: Optional[int] = Field(
        default=None,
        description="Page number in the source document"
    )

    model_config = ConfigDict(extra="forbid")


class SourceMetadata(BaseModel):
    """
    Metadata describing where the document came from.
    """
    source_type: SourceType = Field(
        ...,
        description="Type of the source document"
    )
    source_uri: Optional[str] = Field(
        default=None,
        description="Path or URL of the source document"
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Original file name if available"
    )

    model_config = ConfigDict(extra="forbid")


class BlockMetadata(BaseModel):
    """
    Metadata attached to each extracted content block.
    """
    page: Optional[PageMetadata] = Field(
        default=None,
        description="Page information if applicable"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Loader-specific metadata (kept explicit)"
    )

    model_config = ConfigDict(extra="forbid")


# ============================================================
# Core content models
# ============================================================

class ContentBlock(BaseModel):
    """
    Atomic unit of extracted content.
    This is the smallest retrievable unit before chunking.
    """
    block_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the content block"
    )
    content_type: ContentType = Field(
        ...,
        description="Type of content in this block"
    )
    text: str = Field(
        ...,
        description="Raw extracted text (no summarization or paraphrasing)"
    )
    metadata: BlockMetadata = Field(
        default_factory=BlockMetadata,
        description="Metadata associated with this block"
    )

    model_config = ConfigDict(extra="forbid")


class IngestedDocument(BaseModel):
    """
    Final output of Phase-1 ingestion.
    """
    document_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the ingested document"
    )
    source: SourceMetadata = Field(
        ...,
        description="Source information for this document"
    )
    blocks: List[ContentBlock] = Field(
        ...,
        description="Ordered list of extracted content blocks"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when ingestion occurred"
    )

    model_config = ConfigDict(extra="forbid")