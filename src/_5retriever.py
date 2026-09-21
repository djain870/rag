# from sklearn.metrics.pairwise import cosine_similarity
from ._1config import QDRANT_COLLECTION, TOP_K
from qdrant_client import QdrantClient

import os

def search_enterprise_knowledge(query_vector: str):

    # Using query_points - the modern standard for Qdrant
    client = QdrantClient(url = os.getenv("QDRANT_CLUSTER_ENDPOINT"),api_key = os.getenv("QDRANT_API_KEY"))
    client = QdrantClient(
        url = os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )
    
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit = TOP_K,
        with_payload=True,  # JSON
    )

    results = []
    for res in response.points:
        results.append(
            {"content": res.payload.get("text", ""), "score": res.score}
        )
    return results



if __name__ == "__main__":
    print('hello')