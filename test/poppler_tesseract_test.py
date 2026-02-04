from pdf2image import convert_from_path
import pytesseract
from PIL import Image

POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

pages = convert_from_path(
    "test.pdf",
    poppler_path=POPPLER_PATH,
    dpi=200
)

for i, page in enumerate(pages[:1]):  # just first page
    text = pytesseract.image_to_string(page)
    print(f"\n--- Page {i+1} OCR ---")
    print(text.strip())