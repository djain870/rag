from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

class IngestResponse(BaseModel):
    new_pdfs: int
    changed_pdfs: int
    skipped_pdfs: int
    total_pdfs: int

