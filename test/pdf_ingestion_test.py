from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load environment BEFORE importing ingestion code
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.pdf_ingestion import ingest_pdf
from schema.ingestion import ContentType


def main():
    pdf_path = PROJECT_ROOT / "test" / "test.pdf"

    doc = ingest_pdf(pdf_path)

    print(f"\nDocument ID: {doc.document_id}")
    print(f"Source: {doc.source.file_name}")
    print(f"Blocks extracted: {len(doc.blocks)}\n")

    for block in doc.blocks:
        page = block.metadata.page.page_number if block.metadata.page else None
        print(f"[{block.content_type}] page={page}")
        print(block.text[:200])
        print("-" * 60)


if __name__ == "__main__":
    main()