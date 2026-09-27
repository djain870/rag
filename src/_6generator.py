from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from langsmith import traceable

from ._5retriever import search_enterprise_knowledge
from ._9logger import logger
import asyncio

load_dotenv()
client = AsyncOpenAI()

def build_context(query_embedding: list[float]) -> str:
    """Build the context for the LLM based on the query."""
    # This function would typically search your vector database or knowledge base
    # to find relevant information based on the query.
    # For demonstration purposes, we'll return a placeholder context.
    results = search_enterprise_knowledge(query_embedding)
    # context = "  ".join([results[i]['text'] for i in range(len(results))])
    return results

def build_prompt(query: str, reranked_docs: list[dict]) -> str:
    """Build the prompt sent to the LLM."""

    context = "  ".join([reranked_doc['document'] for reranked_doc in reranked_docs])
    logger.info("Context: %s", context)
    # print(context)

    return f"""
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

@traceable(name="LLM Call")
async def call_llm(prompt: str) -> str:
    """Send a prompt to the LLM."""
    logger.info("Prompt: %s", prompt)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print('hello')