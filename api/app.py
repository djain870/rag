from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Production-ready RAG API"
)

app.include_router(router)