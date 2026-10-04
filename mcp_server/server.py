from langsmith import traceable
from mcp.server.mcpserver import MCPServer

from src.embedding import load_embedding_model
from src.logger import logger
from src.reranker import load_cross_encoder_model
from src.services.rag_service import get_query_response

mcp = MCPServer("RAG Server")
embedding_model = load_embedding_model()
cross_encoder_model = load_cross_encoder_model()

@traceable(name="Search Knowledge")
@mcp.tool()
async def search_knowledge(query: str) -> dict:
    """
    Search the knowledge base for relevant information.
    """
    logger.info("Function called: search_knowledge")
    response = await get_query_response(query)

    return {
        "answer": response["answer"],
        "sources": [
            {
                "source": doc.get("source"),
                "page": doc.get("page"),
                "score": doc.get("score"),
            }
            for doc in response["vector_results"]
        ],
    }

@mcp.tool()
def calculator(a: float, b: float, operation: str) -> float:
    """Perform a basic mathematical calculation."""
    logger.info("Function called: calculator")

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    raise ValueError("Unsupported operation")

if __name__ == "__main__":
    mcp.run()