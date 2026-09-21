from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def build_prompt(query: str, context: list[str]) -> str:
    """Build the prompt sent to the LLM."""

    context_text = "\n\n".join(context)

    return f"""
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context_text}

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