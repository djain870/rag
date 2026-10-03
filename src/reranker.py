import numpy as np
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import sentence_transformers
from langsmith import traceable

from src.config import TOP_K_RERANK
from src.logger import logger

# 1. Initialize the Cross-Encoder model
cross_encoder_model = None

def load_cross_encoder_model():
    """
    Returns the initialized Cross-Encoder model.
    """
    global cross_encoder_model
    if cross_encoder_model is None:
        cross_encoder_model = sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return cross_encoder_model

@traceable(name="Cross Encoder Reranker")
def rerank_documents(query: str, documents: list[str], top_k: int = 3):
    """
    Reranks documents based on their semantic relevance to a query 
    using a Hugging Face CrossEncoder model.
    """
    # 1. Initialize the Cross-Encoder model
    # model = sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 2. Pair the query with each document as required by Cross-Encoders
    # The required format is a list of tuples: [(query, doc1), (query, doc2), ...]
    pairs = [(query, doc) for doc in documents]
    cross_encoder_model = load_cross_encoder_model()
    # 3. Predict similarity scores
    scores = cross_encoder_model.predict(pairs)

    # print(scores)
    # print(type(scores))
    # 4. Sort documents by their scores in descending order
    # argsort returns indices that would sort the array; [::-1] reverses it
    ranked_indices = np.argsort(scores)[::-1]
    
    # 5. Format and filter the top_k results
    ranked_results = []
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        ranked_results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "document": documents[idx]
        })
    # logger.info("Ranked results: %s", ranked_results)
    # print(ranked_results)
    # print(type(ranked_results))
    #return top k reranked results only
    return ranked_results[:TOP_K_RERANK]

if __name__ == "__main__":
    search_query = "How do I fix a deadlock in my application?"
    
    # Simulating broad initial retrieval results (e.g., from BM25 or vector search)
    retrieved_docs = [
        "Advisory locks are acquired explicitly by the application and don't block writes.",
        "Lock contention reduces performance. Contention happens when multiple transactions want a lock.",
        "A thread deadlock in application code happens when two threads permanently block each other.",
        "Two database transactions deadlock when each holds a lock that the other process requires.",
        "To fix a thread deadlock, ensure all threads acquire locks in a strict, uniform order."
    ]
    
    print(f"Query: {search_query}\n")
    print("--- Top Reranked Results ---")
    results = rerank_documents(search_query, retrieved_docs, top_k=3)
    
    for res in results:
        print(f"Rank {res['rank']} [Score: {res['score']:+.4f}]: {res['document']}")