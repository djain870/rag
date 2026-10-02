from pathlib import Path

DATA_DIR = Path("data/true_data")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
TOP_K_RERANK = 3
SCORE_THRESHOLD = 0.7
QDRANT_COLLECTION = "my_rag_project_embeddings"
OPENAI_MODEL = "gpt-4o"