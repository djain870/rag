from pathlib import Path

from src._1config import DATA_DIR
from src._2parser import parse_and_chunk, get_document_hash


docs_path = [f for f in DATA_DIR.glob("**/*.*") if f.suffix in [".pdf", ".pptx", ".docx", ".html", ".txt"]]
file_path = docs_path[0]

doc_hash = get_document_hash(file_path)
assert doc_hash is not None
chunks_meta_list = parse_and_chunk(file_path, doc_hash)
assert len(chunks_meta_list) > 0

