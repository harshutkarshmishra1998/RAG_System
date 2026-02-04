from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

print("POPPLER_PATH =", os.getenv("POPPLER_PATH"))
print("TESSERACT_PATH =", os.getenv("TESSERACT_PATH"))