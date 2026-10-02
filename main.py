import asyncio

from langsmith import traceable

from src.vectordb import initialize_qdrant_collection
from src.logger import logger
from src.services.ingest_documents import process_documents
from src.services.rag_service import get_query_response


@traceable(name="RAG Pipeline")
async def main(query):
    initialize_qdrant_collection()
    process_documents()
    response = await get_query_response(query)
    logger.info("Response: %s", response["answer"])
    return response

if __name__ == "__main__":
    asyncio.run(
        main("What is Kubernetes?")
    )

