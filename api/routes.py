
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

@router.get("/qdrant-health")
def qdrant_health():
    client = QdrantClient(
        url = os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key = os.getenv("QDRANT_API_KEY"),
        timeout = 10000
    )
    result = client.get_collections()
    return {"status": "ok"}


@router.get("/network-test")
def network_test():
    response = requests.get("https://www.google.com", timeout=10)
    return {
        "status": response.status_code
    }

@router.get("/config-test")
def config_test():
    import os

    return {
        "endpoint": os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        "api_key": os.getenv("QDRANT_API_KEY")
    }


import os
import socket
import urllib.request
from fastapi import APIRouter

router = APIRouter()

@router.get("/qdrant-health")
def qdrant_health():
    # 1. Parse host from environment variable
    raw_url = os.getenv("QDRANT_CLUSTER_ENDPOINT", "")
    # Clean up URL to get clean hostname (removes https:// and ports)
    host = raw_url.replace("https://", "").replace("http://", "").split(":")[0].split("/")[0]
    
    diagnostic_results = {
        "configured_url": raw_url,
        "extracted_host": host,
        "step_1_dns_resolution": "Not tested",
        "step_2_raw_tcp_connect": "Not tested",
        "step_3_http_handshake": "Not tested"
    }

    if not host:
        return {"status": "error", "message": "QDRANT_CLUSTER_ENDPOINT env var is missing or empty"}

    # --- STEP 1: TEST DNS RESOLUTION ---
    try:
        ip = socket.gethostbyname(host)
        diagnostic_results["step_1_dns_resolution"] = f"SUCCESS: Resolved to {ip}"
    except Exception as e:
        diagnostic_results["step_1_dns_resolution"] = f"FAILED: {str(e)}"
        return {"status": "network_issue", "details": diagnostic_results}

    # --- STEP 2: TEST RAW PORT CONNECTIVITY ---
    # We will try port 443 (HTTPS) and port 6333 (Default Qdrant HTTP)
    ports_to_test = [443, 6333]
    tcp_statuses = []
    for port in ports_to_test:
        try:
            with socket.create_connection((host, port), timeout=5):
                tcp_statuses.append(f"Port {port}: OPEN")
        except Exception as e:
            tcp_statuses.append(f"Port {port}: CLOSED/TIMEOUT ({str(e)})")
    
    diagnostic_results["step_2_raw_tcp_connect"] = " | ".join(tcp_statuses)

    # --- STEP 3: TEST SIMPLE HTTP GET ---
    try:
        # Try to ping Qdrant's native health endpoint via standard python urllib
        # Using a short 5-second timeout so the app doesn't hang forever
        req_url = f"https://{host}/health" if "https" in raw_url else f"http://{host}:6333/health"
        with urllib.request.urlopen(req_url, timeout=5) as response:
            diagnostic_results["step_3_http_handshake"] = f"SUCCESS: Status {response.getcode()}"
    except Exception as e:
        diagnostic_results["step_3_http_handshake"] = f"FAILED: {str(e)}"

    return {"status": "diagnostic_complete", "details": diagnostic_results}

# @router.get("/sleep")
# async def sleep():
#     # print("started")
#     logger.info("Started at %s", datetime.now().strftime("%H:%M:%S"))
#     await asyncio.sleep(5)
#     # print("Finished")
#     logger.info("Finished at %s", datetime.now().strftime("%H:%M:%S"))
#     return {"message": "Done"}