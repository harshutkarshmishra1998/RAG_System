from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load environment (symmetry with other ingestion tests)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.yt_ingestion import ingest_youtube
from schema.ingestion import ContentType


def main():
    # Replace with a real video you control or trust
    youtube_url = "https://www.youtube.com/watch?v=sDv4f4s2SB8"

    doc = ingest_youtube(youtube_url)

    print("\n================ YOUTUBE INGESTION TEST ================")
    print(f"Document ID   : {doc.document_id}")
    print(f"Source URL    : {doc.source.source_uri}")
    print(f"Blocks count  : {len(doc.blocks)}")
    print("========================================================\n")

    # Print first few blocks for inspection
    for i, block in enumerate(doc.blocks[:10], start=1):
        meta = block.metadata.extra

        print(f"[Block {i}]")
        print(f"Type      : {block.content_type}")
        print(f"Start     : {meta.get('start_time')}")
        print(f"Duration  : {meta.get('duration')}")
        print(f"Source    : {meta.get('source')}")
        print("Text:")
        print(block.text)
        print("-" * 60)

    # Basic sanity assertions (manual, not pytest)
    assert doc.blocks, "No transcript blocks extracted"
    assert all(
        b.content_type == ContentType.TEXT for b in doc.blocks
    ), "Non-TEXT block found in YouTube ingestion"


if __name__ == "__main__":
    main()

# from youtube_transcript_api import YouTubeTranscriptApi
# print(dir(YouTubeTranscriptApi))