from fastapi import APIRouter
from api.models import ChatResponse, ChatRequest, IngestResponse
from src.services.rag_service import get_query_response
from src.services.ingest_documents import process_documents
import time
from datetime import datetime
import asyncio

from src._9logger import logger

router = APIRouter()



@router.get("/health")
def health():
    return {
        "status": "healthy"
    }

@router.get("/chat")
def chat():
    return {
        "status": "chat"
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