from pathlib import Path
from pypdf import PdfReader
import re


def get_first_pdf(data_dir: Path) -> Path:
    """Return the first PDF in the folder, sorted by filename."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    print(pdf_paths[0])
    return pdf_paths[0]


def parse_pdf(pdf_path: Path) -> str:
    """Extract text from every page in a PDF."""
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n\n--- Page {page_number} ---\n{text}")

    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Collapse repeated whitespace so the chunker sees normal text."""
    return re.sub(r"\s+", " ", text).strip()