
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



# @router.get("/sleep")
# async def sleep():
#     # print("started")
#     logger.info("Started at %s", datetime.now().strftime("%H:%M:%S"))
#     await asyncio.sleep(5)
#     # print("Finished")
#     logger.info("Finished at %s", datetime.now().strftime("%H:%M:%S"))
#     return {"message": "Done"}