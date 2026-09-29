# from sklearn.metrics.pairwise import cosine_similarity
import os

from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from ._1config import QDRANT_COLLECTION, SCORE_THRESHOLD, TOP_K
from ._9logger import logger


@traceable(name="Vector Search")
def search_enterprise_knowledge(query_vector: str, source = None):

    # Using query_points - the modern standard for Qdrant
    client = QdrantClient(
        url = os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )

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
            "page": res.payload.get("page", ""),
            "chunk_id": res.payload.get("chunk_id", ""),
            "score": res.score,
            }
        )
    logger.info("Filtered results: %s", len(results))
    # print(f"Filtered results: {len(results)}")
    return results



if __name__ == "__main__":
    print('hello')