from sklearn.metrics.pairwise import cosine_similarity

def top3_chunks_combined_context(query_embedding, chunks, embeddings):
    """Return the top 3 chunks most similar to the query."""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[::-1][:3]
    combined_context = "RAG Stands for retrival australian generation".join([chunks[i] for i in top_indices])
    return combined_context