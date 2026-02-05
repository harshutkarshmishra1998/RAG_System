from schema.ingestion import IngestedDocument
from chunking.block_aware_chunker import block_aware_chunking
from chunking.adaptive_chunk_assembly import adaptive_chunk_assembly
from chunking.chunk_metadata_finalization import (
    finalize_chunk_metadata,
)

# Phase 3_1 was chunking schema definition and metadata finalization, which are now in separate files for better modularity and testability. 
# The execution wrappers for these phases are defined below for integration into the main pipeline.

def run_phase_3_2_block_aware_chunking(
    doc: IngestedDocument,
):
    """
    Phase-3.2 execution wrapper
    """
    return block_aware_chunking(
        blocks=doc.blocks,
        document_id=doc.document_id,
    )

def run_phase_3_3_adaptive_chunk_assembly(chunks):
    """
    Phase-3.3 execution wrapper
    """
    return adaptive_chunk_assembly(chunks)

def run_phase_3_4_chunk_metadata_finalization(chunks):
    """
    Phase-3.4 execution wrapper
    """
    return finalize_chunk_metadata(chunks)