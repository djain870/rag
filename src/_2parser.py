import hashlib
import re
from pathlib import Path

from langsmith import traceable
from pypdf import PdfReader

from ._1config import DATA_DIR
from ._9logger import logger


def get_document_hash(pdf_path: Path) -> str:
    """Return a hash of the PDF contents."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_first_pdf(data_dir: Path) -> Path:
    """Return the first PDF in the folder, sorted by filename."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    logger.info("First PDF found: %s", pdf_paths[0])
    return pdf_paths[0]

def get_all_pdfs(data_dir: Path) -> list[Path]:
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    if not pdf_paths:
        logger.exception("No PDF files found in %s", data_dir)
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    return pdf_paths

@traceable
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
