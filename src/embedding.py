from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    """Generate embeddings for each chunk using the embedding model."""
    return embedding_model.encode(chunks)

def embed_query(query: str):
    """Generate embedding for a query using the embedding model."""
    return embedding_model.encode([query])[0]
