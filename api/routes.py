

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request

from api.models import ChatRequest, ChatResponse, IngestResponse
from src.cache import get_cached_response, save_cached_response
from src.conversation_memory import get_conversation_history, save_conversation_history
from src.query_rephraser import rewrite_query
from src.rate_limiter import check_rate_limit
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

    # Check rate limit for the client
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )

    # Load conversation history
    history = get_conversation_history(chat_request.conversation_id)

    # Rewrite the query based on conversation history
    rephrased_query = rewrite_query(chat_request.query, history)

    cached_response = get_cached_response(
        chat_request.conversation_id,
        rephrased_query
    )

    if cached_response:
        sources = [
            {
                "source": doc.get("source"),
                "page": doc.get("page"),
                "score": doc.get("score"),
            }
            for doc in cached_response["vector_results"]
        ]
        return ChatResponse(
            answer=cached_response["answer"],
            sources=sources,
        )

    response = await get_query_response(rephrased_query)
    answer = response["answer"]

    sources = [
        {
            "source": doc.get("source"),
            "page": doc.get("page"),
            "score": doc.get("score"),
        }
        for doc in response["vector_results"]
    ]

    # Save the conversation history
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
    )

@router.get("/ingest", response_model=IngestResponse)
def ingest():
    return process_documents()
