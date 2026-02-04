from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.web_ingestion import ingest_web_page


def main():
    url = "https://en.wikipedia.org/wiki/Gradient_descent"

    doc = ingest_web_page(url)

    print("\n================ WEB INGESTION TEST =================")
    print(f"Source URL   : {doc.source.source_uri}")
    print(f"Blocks count : {len(doc.blocks)}")
    print("=====================================================\n")

    for block in doc.blocks[:15]:
        print(f"[{block.content_type}]")
        if block.text:
            print(block.text[:200])
        print(block.metadata.extra)
        print("-" * 60)


if __name__ == "__main__":
    main()