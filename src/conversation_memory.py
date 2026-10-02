import json

from src.redis_client import redis_client


MEMORY_TTL = 3600


def get_conversation_history(conversation_id: str) -> list:
    key = f"conversation:{conversation_id}"

    history = redis_client.get(key)

    if not history:
        return []

    return json.loads(history)


def save_conversation_history(
    conversation_id: str,
    history: list,
) -> None:
    key = f"conversation:{conversation_id}"

    redis_client.set(
        key,
        json.dumps(history),
        ex=MEMORY_TTL,
    )