from langsmith import traceable

from src._9logger import logger
from src.services.ingest_documents import process_documents
from src.services.rag_service import get_query_response


@traceable(name="RAG Pipeline")
def main(query):
    process_documents()
    response = get_query_response(query)
    logger.info("Response: %s", response["answer"])
    return response

if __name__ == "__main__":
    main("Who authored document titled 100G Networking technology overview")

