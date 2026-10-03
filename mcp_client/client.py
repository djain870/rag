import asyncio
import json

from mcp import Client, StdioServerParameters
from openai import OpenAI
from src.services.call_llm import call_llm


client = OpenAI()

server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp_server.server"],
)


def convert_mcp_tools_to_openai(mcp_tools):
    """Convert MCP tool definitions to OpenAI tool definitions."""

    tools = []

    for tool in mcp_tools:

        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        })

    return tools


async def run_agent(user_query: str):

    async with Client(server_params) as mcp_client:

        # 1. Discover MCP tools
        tool_result = await mcp_client.list_tools()

        tools = convert_mcp_tools_to_openai(
            tool_result.tools
        )

        print("Available tools:")

        for tool in tools:
            print(
                f"- {tool['function']['name']}"
            )

        # 2. Send question + tools to LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. "
                    "Use available tools when they "
                    "are useful for answering the user."
                ),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        response = await call_llm(
            message=messages,
            tools=tools,
            tool_choice="auto",
        )



        assistant_message = response.choices[0].message

        # 3. No tool required
        if not assistant_message.tool_calls:

            return assistant_message.content

        # 4. Add assistant tool-call message
        messages.append(
            assistant_message
        )

        # 5. Execute requested tools
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(
                f"\nAgent calling tool: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # 6. Call MCP tool
            result = await mcp_client.call_tool(
                tool_name,
                arguments,
            )

            # Extract text from MCP result
            tool_output = "\n".join(
                item.text
                for item in result.content
                if hasattr(item, "text")
            )

            # 7. Give tool result back to LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output,
            })

        # 8. Ask LLM for final answer
        final_response = await call_llm(
            message=messages,
        )

        return final_response.choices[0].message.content


if __name__ == "__main__":

    answer = asyncio.run(
        run_agent(
            "What is Kubernetes?"
        )
    )

    print("\nFinal Answer:")
    print(answer)