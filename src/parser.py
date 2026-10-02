import hashlib
import re
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langsmith import traceable
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .config import DATA_DIR
from .logger import logger

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print(embedding_model.max_seq_length)
converter = DocumentConverter()
chunker = HybridChunker(
    tokenizer = embedding_model.tokenizer,
    max_tokens = 200,
)
print(chunker)
print("Chunker max tokens:", chunker.max_tokens)
# def get_document_hash(pdf_path: Path) -> str:
#     """Return a hash of the PDF contents."""
#     sha256 = hashlib.sha256()
#     with open(pdf_path, "rb") as f:
#         while chunk := f.read(8192):
#             sha256.update(chunk)
#     return sha256.hexdigest()

def get_document_hash(doc_path: Path) -> str:
    """Return a hash of the PDF contents."""
    sha256 = hashlib.sha256()
    with open(doc_path, "rb") as f:
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

@traceable
def parse_document(file_path: Path) -> str:
    """
    Convert a supported document into Markdown text using Docling.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)

    return result.document



def parse_and_chunk(file_path: Path, document_hash: str):
    result = converter.convert(file_path)

    chunks = list(chunker.chunk(result.document))

    return [
        {
            "text": chunk.text,
            "metadata": extract_chunk_metadata(chunk, document_hash),
        }
        for chunk in chunks
    ]

def extract_chunk_metadata(chunk, document_hash):
    meta = chunk.meta

    page = None

    if meta.doc_items:
        provenance = meta.doc_items[0].prov

        if provenance:
            page = provenance[0].page_no

    return {
        "source": meta.origin.filename if meta.origin else None,
        "document_type": meta.origin.mimetype if meta.origin else None,
        "document_hash": document_hash,
        "headings": meta.headings,
        "page": page,
    }

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
    # print(DATA_DIR)
    # DATA_DIR = Path("../data/noisy_data")
    # print(DATA_DIR)
    pdf_path = get_first_pdf(DATA_DIR)
    # print(DATA_DIR)
    print(parse_pdf(pdf_path))
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    # print(result.document.export_to_markdown())
    result.document.save_as_markdown("test.md")
