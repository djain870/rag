import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langsmith import traceable

from src.config import OPENAI_MODEL
from src.logger import logger

load_dotenv()

model = None

def get_model():
    global model

    if model is None:
        model = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )

    return model

mcp_tools = None

async def start_mcp():
    global mcp_adapter, mcp_tools

    mcp_adapter = MCPAdapter(
        {
            "mcpServers": {
                "rag": {
                    "command": "python",
                    "args": ["-m", "mcp_server.server"],
                    "transport": "stdio",
                }
            }
        }
    )

    await mcp_adapter.__aenter__()
    mcp_tools = await list_mcp_tools(mcp_adapter)

async def stop_mcp():
    global mcp_adapter

    if mcp_adapter is not None:
        await mcp_adapter.__aexit__(None, None, None)
        mcp_adapter = None

@traceable(name="List MCP Tools")
async def list_mcp_tools(adapter):
    return await adapter.list_tools()


@traceable(name="Create Agent")
def create_rag_agent(model, tools):
    return create_agent(
        get_model(),
        tools,
    )

@traceable(name="Run Agent")
async def run_agent(user_query: str):
    logger.info("Function called: run agent")
    tools = mcp_tools

    logger.info(
        "Available tools: %s",
        [tool.name for tool in tools],
    )

    for tool in tools:
        logger.info("- %s", tool.name)

    agent = create_rag_agent(
        model,
        tools,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        }
    )
    tools_used = [
        tool_call["name"]
        for message in result["messages"]
        for tool_call in getattr(message, "tool_calls", [])
    ]

    logger.info("Tools used: %s", tools_used)
    messages = result["messages"]

    answer = messages[-1].content

    sources = []
    # logger.info("Messages: %s", messages)

    for message in messages:
        if (
            getattr(message, "type", None) == "tool"
            and isinstance(message.content, list)
        ):
            for content in message.content:
                if (
                    isinstance(content, dict)
                    and content.get("type") == "text"
                ):
                    tool_result = json.loads(content["text"])
                    if isinstance(tool_result, dict):
                        sources.extend(
                            tool_result.get("sources", [])
                        )

    return {
        "answer": answer,
        "sources": sources,
        "tools_used": tools_used,
    }

if __name__ == "__main__":
    import asyncio

    answer = asyncio.run(
        run_agent(
            "What is Kubernetes and what is 125 × 48?"
        )
    )

    print("\nFinal Answer:")
    print(answer)