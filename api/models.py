from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    conversation_id: str

class Source(BaseModel):
    source: str | None = None
    page: int | None = None
    score: float | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    tools_used: list[str] = []


class IngestResponse(BaseModel):
    new_pdfs: int
    changed_pdfs: int
    skipped_pdfs: int
    total_pdfs: int

