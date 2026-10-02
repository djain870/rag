import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import QDRANT_COLLECTION
from .logger import logger

load_dotenv()

def qdrant_client():
    client = QdrantClient(
        url = os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key = os.getenv("QDRANT_API_KEY"),
        port = 443,
        https = True,
        timeout = 30,
    )
    return client

def retrieve_document_hash(source: str) -> str:
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
    response = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=query_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if len(response[0]) == 0:
        return None

    return response[0][0].payload.get("document_hash")

def delete_document(source: str) -> str:
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
    response = client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=query_filter,
        ),
    )
    print(response)
    print(response.status)
    print(type(response.status))
    return response.status

def check_document_exists(source: str) -> bool:
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
    response = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=query_filter,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    return len(response[0]) > 0



def initialize_qdrant_collection():
    """
    Initializes a Qdrant collection for storing embeddings.
    """
    # 1. Initialize the client (Using a local file for persistence)
    client = qdrant_client()

    # 2. Create a collection (Specify the size of your embeddings and distance metric)
    # 3. if colletion already exists, check below for exception
    if QDRANT_COLLECTION in [col.name for col in client.get_collections().collections]:
        print("Collection already exists. Skipping creation.")
        return client, QDRANT_COLLECTION
    else:
        logger.info("Creating new collection for embeddings.")
        # print("Creating new collection for embeddings.")
        collection_name = QDRANT_COLLECTION
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # e.g., OpenAI size
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="source",
            field_schema="keyword",
        )
        return client, collection_name

def upsert_embeddings_to_qdrant(embeddings, chunks):
    """
    Upserts embeddings and their corresponding metadata into the Qdrant collection.
    """
    client, collection_name = initialize_qdrant_collection()

    # 3. Save your embeddings as "Points"
    points = [
        PointStruct(
            id=str(uuid.uuid4()), 
            vector=vector, 
            payload={
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "document_type": chunk["metadata"]["document_type"],
                "document_hash": chunk["metadata"]["document_hash"],
                "chunk_index": index,
                "page": chunk["metadata"]["page"],
                "headings": chunk["metadata"]["headings"],
            },
        )
        for index, (vector, chunk) in enumerate(zip(embeddings, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)

if __name__ == "__main__":
    print("hello")
    # print("Embeddings generated for all chunks.")
