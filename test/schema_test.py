from schema.ingestion import (
    IngestedDocument,
    ContentBlock,
    ContentType,
    SourceMetadata,
    SourceType,
)

doc = IngestedDocument(
    source=SourceMetadata(
        source_type=SourceType.PDF,
        source_uri="test.pdf",
        file_name="test.pdf",
    ),
    blocks=[
        ContentBlock(
            content_type=ContentType.TEXT,
            text="Schema test successful"
        )
    ],
)

print(doc.model_dump_json(indent=2))