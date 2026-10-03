from mcp.server.mcpserver import MCPServer, Context

from src.services.rag_service import get_query_response


mcp = MCPServer("RAG Server")


@mcp.tool()
async def search_knowledge(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    """

    response = await get_query_response(query)

    return response["answer"]


if __name__ == "__main__":
    mcp.run()