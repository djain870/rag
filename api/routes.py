
from fastapi import APIRouter
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import requests

from api.models import ChatRequest, ChatResponse, IngestResponse
from src.services.ingest_documents import process_documents
from src.services.rag_service import get_query_response

router = APIRouter()

load_dotenv()

@router.get("/health")
def health():
    return {
        "status": "healthy"
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return ChatResponse(
        answer= (await get_query_response(request.query))['answer']
    )

@router.get("/ingest", response_model=IngestResponse)
def ingest():
    return process_documents()



import os
import socket
import urllib.request
from fastapi import APIRouter


@router.get("/qdrant-health")
def qdrant_health():
    # Force the client to use port 443 and explicit HTTPS settings
    client = QdrantClient(
        url="https://10d22751-c9d1-4106-81de-93303760d055.australia-southeast1-0.gcp.cloud.qdrant.io",
        port=443,               # Force port 443 explicitly
        https=True,             # Force HTTPS
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=30              # 30 seconds is plenty now that the port is correct
    )
    
    try:
        # This will now route through the open 443 port with authentication
        result = client.get_collections()
        return {"status": "ok", "collections": [c.name for c in result.collections]}
    except Exception as e:
        return {"status": "client_error", "error": str(e)}


# @router.get("/sleep")
# async def sleep():
#     # print("started")
#     logger.info("Started at %s", datetime.now().strftime("%H:%M:%S"))
#     await asyncio.sleep(5)
#     # print("Finished")
#     logger.info("Finished at %s", datetime.now().strftime("%H:%M:%S"))
#     return {"message": "Done"}