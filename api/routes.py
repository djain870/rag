

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from src.rate_limiter import check_rate_limit

from api.models import ChatRequest, ChatResponse, IngestResponse
from src.services.ingest_documents import process_documents
from src.services.rag_service import get_query_response

router = APIRouter()

load_dotenv()

@router.get("/health")
def health(request: Request):
    client_id = request.client.host

    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )
    return {
        "status": "healthy"
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):

    client_id = request.client.host

    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )

    response = await get_query_response(chat_request.query)

    return ChatResponse(
        answer=response["answer"]
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