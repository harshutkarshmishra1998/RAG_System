from pathlib import Path
from dotenv import load_dotenv
import shutil

# ------------------------------------------------------------
# Load environment (symmetry with other tests)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from schema.google_drive_loader import download_from_google_drive
from schema.pdf_ingestion import ingest_pdf
from schema.docx_ingestion import ingest_docx
from schema.txt_ingestion import ingest_txt


def main():
    """
    End-to-end test:
    Google Drive link → download → type detection → ingestion
    """

    # ⚠️ IMPORTANT:
    # Use a PUBLIC Google Drive file link you control
    drive_url = (
        # "https://docs.google.com/document/d/1Z-E-_Ab_F98Wy6bwfGawy3RIYKjm1kq0/edit?usp=sharing&ouid=105012101660355939612&rtpof=true&sd=true"
        "https://drive.google.com/file/d/1Gz9wWwBrTjssEoWJNu5HVenhUfYZTxbh/view?usp=sharing"
    )

    tmp_dir = PROJECT_ROOT / "test" / "_tmp_drive"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\n================ GOOGLE DRIVE INGESTION TEST ================")
    print("Downloading file from Google Drive...")

    local_file = download_from_google_drive(drive_url, tmp_dir)

    print(f"Downloaded to: {local_file}")
    print(f"File suffix  : {local_file.suffix}")

    # ------------------------------------------------------------
    # Dispatch based on file extension
    # (Unified router will replace this later)
    # ------------------------------------------------------------
    if local_file.suffix.lower() == ".pdf":
        doc = ingest_pdf(local_file)

    elif local_file.suffix.lower() == ".docx":
        doc = ingest_docx(local_file)

    elif local_file.suffix.lower() == ".txt":
        doc = ingest_txt(local_file)

    else:
        raise RuntimeError(f"Unsupported file type: {local_file.suffix}")

    print("\n================ INGESTION RESULT ===========================")
    print(f"Document ID  : {doc.document_id}")
    print(f"Source URI   : {doc.source.source_uri}")
    print(f"File name   : {doc.source.file_name}")
    print(f"Blocks count: {len(doc.blocks)}")
    print("============================================================\n")

    # Print a few blocks for sanity
    for block in doc.blocks[:5]:
        print(f"[{block.content_type}]")
        print(block.text[:200])
        print("-" * 60)

    assert doc.blocks, "No blocks extracted from Google Drive file"


if __name__ == "__main__":
    main()