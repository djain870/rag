from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from mcp_client.client import start_mcp, stop_mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_mcp()
    yield
    await stop_mcp()

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Production-ready RAG API",
    lifespan=lifespan,
)

app.include_router(router)