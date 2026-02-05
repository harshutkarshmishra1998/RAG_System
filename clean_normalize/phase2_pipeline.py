# clean-normalize/phase2_pipeline.py

from schema.ingestion import IngestedDocument
from clean_normalize.canonical_text_sanitation import (
    canonical_text_sanitation,
)
from clean_normalize.structural_block_segmentation import (
    structural_block_segmentation,
)
from clean_normalize.boilerplate_detection_suppression import (
    boilerplate_detection_suppression,
)
from clean_normalize.table_normalization import (
    table_normalization,
)
from clean_normalize.metadata_canonicalization import (
    metadata_canonicalization,
)
from clean_normalize.deterministic_ordering_hashing import (
    deterministic_ordering_and_hashing,
)


def run_phase_2_1_canonical_text_sanitation(
    doc: IngestedDocument,
) -> IngestedDocument:
    """
    Phase-2.1 execution wrapper.
    """
    for block in doc.blocks:
        if block.text:
            block.text = canonical_text_sanitation(block.text)

    return doc

def run_phase_2_2_structural_block_segmentation(
    doc: IngestedDocument,
) -> IngestedDocument:
    """
    Phase-2.2 execution wrapper
    """
    doc.blocks = structural_block_segmentation(doc.blocks)
    return doc

def run_phase_2_3_boilerplate_detection_suppression(
    doc: IngestedDocument,
) -> IngestedDocument:
    """
    Phase-2.3 execution wrapper
    """
    doc.blocks = boilerplate_detection_suppression(doc.blocks)
    return doc

def run_phase_2_4_table_normalization(
    doc: IngestedDocument,
) -> IngestedDocument:
    """
    Phase-2.4 execution wrapper
    """
    doc.blocks = table_normalization(doc.blocks)
    return doc

def run_phase_2_5_metadata_canonicalization(
    doc: IngestedDocument,
) -> IngestedDocument:
    """
    Phase-2.5 execution wrapper
    """
    doc.blocks = metadata_canonicalization(
        doc.blocks,
        source_type=doc.source.source_type,
    )
    return doc

def run_phase_2_6_deterministic_ordering_hashing(
    doc: IngestedDocument,
) -> IngestedDocument:
    """
    Phase-2.6 execution wrapper
    """
    doc.blocks = deterministic_ordering_and_hashing(doc.blocks)
    return doc