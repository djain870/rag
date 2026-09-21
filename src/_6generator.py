from openai import OpenAI
from dotenv import load_dotenv
from ._5retriever import search_enterprise_knowledge

load_dotenv()
client = OpenAI()

def build_context(query_embedding):
    """Build the context for the LLM based on the query."""
    # This function would typically search your vector database or knowledge base
    # to find relevant information based on the query.
    # For demonstration purposes, we'll return a placeholder context.
    results = search_enterprise_knowledge(query_embedding)
    context = "  ".join([results[i]['text'] for i in range(len(results))])
    return context

def build_prompt(query: str, context: list[str]) -> str:
    """Build the prompt sent to the LLM."""

    # context_text = "\n\n".join(context)

    return f"""
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{query}
"""

def call_llm(prompt: str) -> str:
    """Send a prompt to the LLM."""

    response = client.chat.completions.create(
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