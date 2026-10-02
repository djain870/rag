# from sklearn.metrics.pairwise import cosine_similarity

from langsmith import traceable
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import QDRANT_COLLECTION, SCORE_THRESHOLD, TOP_K
from .logger import logger
from .vectordb import qdrant_client


@traceable(name="Vector Search")
def search_enterprise_knowledge(query_vector: str, source = None):

    # Using query_points - the modern standard for Qdrant
    client = qdrant_client()

    query_filter = None
    if source:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        )
    
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter= query_filter,
        limit = TOP_K,
        with_payload=True,  # JSON
    )

    results = []
    logger.info("Total results: %s", len(response.points))
    # print(f"Total results: {len(response.points)}")
    for res in response.points:
        if res.score < SCORE_THRESHOLD:
            continue
        results.append({
            "text": res.payload.get("text", ""),
            "source": res.payload.get("source", ""),
            "document_type": res.payload.get("document_type", ""),
            "document_hash": res.payload.get("document_hash", ""),
            "page": res.payload.get("page", ""),
            "headings": res.payload.get("headings", ""),
            "chunk_index": res.payload.get("chunk_index", ""),
            "score": res.score,
            }
        )
    logger.info("Filtered results: %s", len(results))
    # print(f"Filtered results: {len(results)}")
    return results



if __name__ == "__main__":
    print('hello')