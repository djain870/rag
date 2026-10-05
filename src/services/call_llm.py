from dotenv import load_dotenv
from langsmith import traceable
from openai import AsyncOpenAI

from src.config import OPENAI_MODEL

load_dotenv()

client = None
def get_client():
    global client

    if client is None:
        client = AsyncOpenAI()

    return client

@traceable(name="LLM Call")
async def call_llm(message: list, tools: list | None = None, tool_choice: list | None = None) -> list:
    """Send message to the LLM."""
    client = get_client()
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=message,
        temperature=0,
        tools=tools,
        tool_choice=tool_choice,
    )

    return response