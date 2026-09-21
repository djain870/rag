from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str])-> list[list[float]]:
    """Generate embeddings for each chunk using the embedding model."""
    embeddings = embedding_model.encode(chunks)
    # upsert_embeddings_to_qdrant(embeddings, chunks)
    return embeddings

def embed_query(query: str):
    """Generate embedding for a query using the embedding model."""
    return embedding_model.encode([query])[0]
