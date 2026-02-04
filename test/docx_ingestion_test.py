from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.docx_ingestion import ingest_docx
from schema.ingestion import ContentType


def main():
    docx_path = PROJECT_ROOT / "test" / "test.docx"

    doc = ingest_docx(docx_path)

    print(f"\nDocument ID: {doc.document_id}")
    print(f"Source: {doc.source.file_name}")
    print(f"Blocks extracted: {len(doc.blocks)}\n")

    for block in doc.blocks:
        print(f"[{block.content_type}]")
        print(block.text[:200])
        print("-" * 60)


if __name__ == "__main__":
    main()