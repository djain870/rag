import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL
from src.logger import logger

load_dotenv()

model = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
)


async def run_agent(user_query: str):
    logger.info("Function called: run agent")
    async with MCPAdapter(
        {
            "mcpServers": {
                "rag": {
                    "command": "python",
                    "args": ["-m", "mcp_server.server"],
                    "transport": "stdio",
                }
            }
        }
    ) as adapter:

        tools = await adapter.list_tools()

        print("Available tools:")

        for tool in tools:
            print(f"- {tool.name}")

        agent = create_agent(
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