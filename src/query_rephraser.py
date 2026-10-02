from openai import OpenAI

from .config import OPENAI_MODEL

client = OpenAI()


def rewrite_query(
    query: str,
    history: list,
) -> str:

    if not history:
        return query

    conversation = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-6:]
    )

    prompt = f"""
Rewrite the user's latest question into a standalone search query.

Use the conversation history to resolve references such as:
- it
- they
- this
- that
- the above
- previous question

Do not answer the question.

Return only the rewritten query.

Conversation:
{conversation}

Latest question:
{query}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()