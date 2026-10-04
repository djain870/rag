

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request

from api.models import ChatRequest, ChatResponse, IngestResponse
from mcp_client.client import run_agent
from src.cache import get_cached_response, save_cached_response
from src.conversation_memory import get_conversation_history, save_conversation_history
from src.logger import logger
from src.query_rephraser import rewrite_query
from src.rate_limiter import check_rate_limit
from src.services.ingest_documents import process_documents

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

    history = get_conversation_history(
        chat_request.conversation_id
    )

    rephrased_query = rewrite_query(
        chat_request.query,
        history,
    )
    logger.info("Rephrased query: %s", rephrased_query)
    cached_response = get_cached_response(
        chat_request.conversation_id,
        rephrased_query,
    )

    if cached_response:
        return ChatResponse(
            answer=cached_response["answer"],
            sources=cached_response["sources"],
            tools_used=cached_response["tools_used"],
        )

    response = await run_agent(rephrased_query)

    answer = response["answer"]
    sources = response["sources"]
    tools_used = response["tools_used"]

    history.append({
        "role": "user",
        "content": chat_request.query,
    })


    history.append({
        "role": "assistant",
        "content": answer,
    })

    save_conversation_history(
        chat_request.conversation_id,
        history,
    )

    save_cached_response(
        chat_request.conversation_id,
        rephrased_query,
        response,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        tools_used=tools_used,
    )

@router.get("/ingest", response_model=IngestResponse)
def ingest():
    return process_documents()
