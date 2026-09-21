from pathlib import Path
from pypdf import PdfReader
import re

from ._1config import DATA_DIR


def get_first_pdf(data_dir: Path) -> Path:
    """Return the first PDF in the folder, sorted by filename."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    print(pdf_paths[0])
    return pdf_paths[0]


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from every page in a PDF."""
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "page": page_number,
            "text": text
        })

    return pages


def clean_text(pages) -> list[dict]:
    """Collapse repeated whitespace so the chunker sees normal text."""
    cleaned_pages = []
    for page in pages:
        cleaned_pages.append({
            "page": page["page"],
            "text": re.sub(r"\s+", " ", page["text"]).strip()
        })

    return cleaned_pages


if __name__ == "__main__":
    pdf_path = get_first_pdf(DATA_DIR)
    print(parse_pdf(pdf_path))
