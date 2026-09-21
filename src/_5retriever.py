# from sklearn.metrics.pairwise import cosine_similarity
from ._1config import QDRANT_COLLECTION, TOP_K
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os

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
    for res in response.points:
        results.append({
            "text": res.payload.get("text", ""),
            "source": res.payload.get("source", ""),
            "page": res.payload.get("page", ""),
            "chunk_id": res.payload.get("chunk_id", ""),
            "score": res.score,
            }
        )
    # print(results)
    return results



if __name__ == "__main__":
    print('hello')