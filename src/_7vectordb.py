from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import os
from dotenv import load_dotenv

load_dotenv()

def initialize_qdrant_collection():
    """
    Initializes a Qdrant collection for storing embeddings.
    """
    # 1. Initialize the client (Using a local file for persistence)
    client = QdrantClient(
        url = os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )

    # 2. Create a collection (Specify the size of your embeddings and distance metric)
    # 3. if colletion already exists, check below for exception
    if "my_rag_project_embeddings" in [col.name for col in client.get_collections().collections]:
        print("Collection already exists. Skipping creation.")
        return client, "my_rag_project_embeddings"
    else:
        print("Creating new collection for embeddings.")
        collection_name = "my_rag_project_embeddings"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # e.g., OpenAI size
        )
        return client, collection_name

def upsert_embeddings_to_qdrant(embeddings, chunks):
    """
    Upserts embeddings and their corresponding metadata into the Qdrant collection.
    """
    client, collection_name = initialize_qdrant_collection()

    # 3. Save your embeddings as "Points"
    points = [
        PointStruct(id=idx, vector=vector, payload=meta)
        for idx, (vector, meta) in enumerate(zip(embeddings, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)

if __name__ == "__main__":
    print("hello")
    # print("Embeddings generated for all chunks.")
