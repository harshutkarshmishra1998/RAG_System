from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.ingest_router import ingest


def run_test(input_value):
    print("\n================ INGEST =================")
    print(f"Input: {input_value}")

    doc = ingest(input_value)

    print(f"Document ID : {doc.document_id}")
    print(f"Source type : {doc.source.source_type}")
    print(f"Source URI  : {doc.source.source_uri}")
    print(f"Blocks      : {len(doc.blocks)}")
    print("========================================")

    assert doc.blocks, "No blocks extracted"


def main():
    tests = [
        PROJECT_ROOT / "test" / "test.pdf",
        PROJECT_ROOT / "test" / "test.docx",
        PROJECT_ROOT / "test" / "test.txt",
        "https://www.youtube.com/watch?v=sDv4f4s2SB8",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit?usp=sharing&ouid=105012101660355939612&rtpof=true&sd=true",
    ]

    for item in tests:
        run_test(item)


if __name__ == "__main__":
    main()