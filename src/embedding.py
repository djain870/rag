import logging

from langsmith import traceable

from src.logger import logger

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from sentence_transformers import SentenceTransformer

embedding_model = None

@traceable(name="Load Embedding Model")
def load_embedding_model():
    """Load the embedding model."""
    logger.info("Function called: load embedding model")
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

def embed_chunks(chunks: list[str])-> list[list[float]]:
    """Generate embeddings for each chunk using the embedding model."""
    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(chunks)
    # upsert_embeddings_to_qdrant(embeddings, chunks)
    return embeddings

@traceable(name="Embed Query")
def embed_query(query: str):
    """Generate embedding for a query using the embedding model."""
    logger.info("Function called: embed query")
    embedding_model = load_embedding_model()
    return embedding_model.encode([query])[0]
