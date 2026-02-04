from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.txt_ingestion import ingest_txt


def main():
    txt_path = PROJECT_ROOT / "test" / "test.txt"

    doc = ingest_txt(txt_path)

    print(f"\nDocument ID: {doc.document_id}")
    print(f"Source: {doc.source.file_name}")
    print(f"Blocks extracted: {len(doc.blocks)}\n")

    for block in doc.blocks[:20]:
        line_no = block.metadata.extra.get("line_number")
        print(f"[TEXT] line={line_no}")
        print(block.text)
        print("-" * 60)


if __name__ == "__main__":
    main()