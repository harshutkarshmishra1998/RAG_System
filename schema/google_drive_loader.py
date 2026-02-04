from __future__ import annotations

import re
import requests
from pathlib import Path


def download_from_google_drive(url: str, dest_dir: Path) -> Path:
    """
    Download a publicly shared Google Drive file.
    Returns path to downloaded file.
    """

    match = re.search(r"/d/([^/]+)", url) or re.search(r"id=([^&]+)", url)
    if not match:
        raise ValueError("Invalid Google Drive file URL")

    file_id = match.group(1)

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    # Try to infer filename
    filename = "drive_file"
    cd = response.headers.get("content-disposition")
    if cd and "filename=" in cd:
        filename = cd.split("filename=")[-1].strip('"')

    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / filename

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return file_path