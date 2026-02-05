from typing import List
from schema.ingestion import ContentBlock, ContentType
from chunking.chunk_schema import Chunk


TARGET_CHARS = 2000     # ~400–500 tokens
MAX_CHARS = 3500
MIN_CHARS = 500


def block_aware_chunking(
    blocks: List[ContentBlock],
    document_id: str,
) -> List[Chunk]:
    """
    Phase-3 block-aware chunking.
    Deterministic. Structure-driven.
    """

    chunks: List[Chunk] = []
    buffer_text = ""
    buffer_blocks = []
    buffer_types = set()
    page_numbers = []

    chunk_index = 0

    def flush():
        nonlocal buffer_text, buffer_blocks, buffer_types, page_numbers, chunk_index

        text = buffer_text.strip()

        if not text:
            buffer_text = ""
            buffer_blocks = []
            buffer_types = set()
            page_numbers = []
            return

        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                block_ids=[b.block_id for b in buffer_blocks],
                content_types=list(buffer_types),
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
            )
        )

        chunk_index += 1
        buffer_text = ""
        buffer_blocks = []
        buffer_types = set()
        page_numbers = []

    for block in blocks:
        meta = block.metadata.extra

        # Skip boilerplate completely
        if meta.get("boilerplate"):
            continue

        # TABLE blocks → standalone chunks
        if block.content_type == ContentType.TABLE:
            flush()
            if not block.text.strip():
                continue
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=block.text.strip(),
                    block_ids=[block.block_id],
                    content_types=[block.content_type.value],
                    page_start=block.metadata.page.page_number if block.metadata.page else None,
                    page_end=block.metadata.page.page_number if block.metadata.page else None,
                )
            )
            chunk_index += 1
            continue

        # IMAGE_OCR blocks → smaller isolated chunks
        if block.content_type == ContentType.IMAGE_OCR:
            flush()
            if not block.text.strip():
                continue
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=block.text.strip(),
                    block_ids=[block.block_id],
                    content_types=[block.content_type.value],
                    page_start=block.metadata.page.page_number if block.metadata.page else None,
                    page_end=block.metadata.page.page_number if block.metadata.page else None,
                    metadata={"low_confidence": True},
                )
            )
            chunk_index += 1
            continue

        # TEXT blocks → buffered chunking
        candidate = buffer_text + "\n\n" + block.text if buffer_text else block.text

        if len(candidate) > MAX_CHARS:
            flush()

        buffer_text = buffer_text + "\n\n" + block.text if buffer_text else block.text
        buffer_blocks.append(block)
        buffer_types.add(block.content_type.value)

        if block.metadata.page and block.metadata.page.page_number:
            page_numbers.append(block.metadata.page.page_number)

        if len(buffer_text) >= TARGET_CHARS:
            flush()

    flush()
    return chunks