from fastapi import APIRouter
from api.models import ChatResponse, ChatRequest, IngestResponse
from src.services.rag_service import get_query_response
from src.services.ingest_documents import process_documents

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
def chat(request: ChatRequest):
    return ChatResponse(
        answer=get_query_response(request.query)['answer']
    )

@router.get("/ingest", response_model=IngestResponse)
def ingest():
    return process_documents()