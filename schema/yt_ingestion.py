from __future__ import annotations

import re
import json
import os
import subprocess
import tempfile
from typing import List

import requests
from youtube_transcript_api import YouTubeTranscriptApi

from schema.ingestion import (
    IngestedDocument,
    ContentBlock,
    ContentType,
    SourceMetadata,
    SourceType,
    BlockMetadata,
)

# ============================================================
# Helpers
# ============================================================

def _extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([^&?/]+)", url)
    if not match:
        raise ValueError(f"Invalid YouTube URL: {url}")
    return match.group(1)


# ============================================================
# YouTube ingestion
# ============================================================

def ingest_youtube(url: str) -> IngestedDocument:
    """
    Bulletproof YouTube transcript ingestion.

    Fallback order:
    1. youtube_transcript_api.get_transcript (PRIMARY)
    2. yt-dlp (if installed)
    3. Invidious API (best-effort)

    Fails only if captions truly do not exist.
    """

    video_id = _extract_video_id(url)
    # print(f"[DEBUG] Extracted video_id = '{video_id}'")

    blocks: List[ContentBlock] = []

    # ========================================================
    # METHOD 1 — youtube_transcript_api (PRIMARY, RELIABLE)
    # ========================================================
    try:
        segments = YouTubeTranscriptApi.get_transcript( #type: ignore
            video_id,
            languages=["en"],
        )

        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            blocks.append(
                ContentBlock(
                    content_type=ContentType.TEXT,
                    text=text,
                    metadata=BlockMetadata(
                        extra={
                            "video_id": video_id,
                            "start_time": seg.get("start"),
                            "duration": seg.get("duration"),
                            "source": "youtube_api",
                        }
                    ),
                )
            )

        if blocks:
            return IngestedDocument(
                source=SourceMetadata(
                    source_type=SourceType.YOUTUBE,
                    source_uri=url,
                    file_name=video_id,
                ),
                blocks=blocks,
            )

    except Exception as e:
        print("[DEBUG] youtube_api failed:", e)

    # ========================================================
    # METHOD 2 — yt-dlp (OPTIONAL, requires binary)
    # ========================================================
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs", "en",
                "--sub-format", "json3",
                "-o", os.path.join(tmpdir, "%(id)s.%(ext)s"),
                url,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                for fname in os.listdir(tmpdir):
                    if fname.endswith(".json3"):
                        with open(os.path.join(tmpdir, fname), encoding="utf-8") as f:
                            data = json.load(f)

                        for event in data.get("events", []):
                            for seg in event.get("segs", []):
                                text = seg.get("utf8", "").strip()
                                if not text:
                                    continue

                                blocks.append(
                                    ContentBlock(
                                        content_type=ContentType.TEXT,
                                        text=text,
                                        metadata=BlockMetadata(
                                            extra={
                                                "video_id": video_id,
                                                "source": "yt_dlp",
                                            }
                                        ),
                                    )
                                )

                if blocks:
                    return IngestedDocument(
                        source=SourceMetadata(
                            source_type=SourceType.YOUTUBE,
                            source_uri=url,
                            file_name=video_id,
                        ),
                        blocks=blocks,
                    )

    except Exception as e:
        print("[DEBUG] yt-dlp failed:", e)

    # ========================================================
    # METHOD 3 — Invidious (BEST-EFFORT)
    # ========================================================
    try:
        api_url = f"https://invidious.fdn.fr/api/v1/captions/{video_id}"
        resp = requests.get(api_url, timeout=15)
        resp.raise_for_status()

        captions = resp.json()
        english = [
            c for c in captions
            if c.get("languageCode", "").startswith("en")
        ]

        if not english:
            raise RuntimeError("No English captions via Invidious")

        xml_resp = requests.get(english[0]["url"], timeout=15)
        xml_resp.raise_for_status()

        text = re.sub(r"<[^>]+>", "", xml_resp.text)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            blocks.append(
                ContentBlock(
                    content_type=ContentType.TEXT,
                    text=text,
                    metadata=BlockMetadata(
                        extra={
                            "video_id": video_id,
                            "source": "invidious",
                        }
                    ),
                )
            )

            return IngestedDocument(
                source=SourceMetadata(
                    source_type=SourceType.YOUTUBE,
                    source_uri=url,
                    file_name=video_id,
                ),
                blocks=blocks,
            )

    except Exception as e:
        print("[DEBUG] invidious failed:", e)

    # ========================================================
    # FINAL FAIL
    # ========================================================
    raise RuntimeError(
        "This video has no accessible captions.\n"
        "All extraction methods failed:\n"
        "- youtube_transcript_api\n"
        "- yt-dlp\n"
        "- Invidious"
    )