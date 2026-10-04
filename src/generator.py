from langsmith import traceable

from src.logger import logger
from src.retriever import search_enterprise_knowledge
from src.services.call_llm import call_llm


@traceable(name="Build Context")
def build_context(query_embedding: list[float]) -> str:
    """Build the context for the LLM based on the query."""
    logger.info("Function called: build context")
    # This function would typically search your vector database or knowledge base
    # to find relevant information based on the query.
    # For demonstration purposes, we'll return a placeholder context.
    results = search_enterprise_knowledge(query_embedding)
    # context = "  ".join([results[i]['text'] for i in range(len(results))])
    return results

@traceable(name="Build Prompt")
def build_prompt(query: str, reranked_docs: list[dict]) -> str:
    """Build the prompt sent to the LLM."""
    logger.info("Function called: build prompt")
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

@traceable(name="Generate Response")
async def generate_response(prompt: str) -> str:
    """Send a prompt to the LLM."""
    # logger.info("Prompt: %s", prompt)
    logger.info("Function called: generate response")
    response = await call_llm(
        message=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print('hello')