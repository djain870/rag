import json

from src.logger import logger
from src.redis_client import USE_REDIS, redis_client

MEMORY_TTL = 3600


def get_conversation_history(conversation_id: str) -> list:
    logger.info("Function called: get conversation history")

    if not USE_REDIS:
        return []

    key = f"conversation:{conversation_id}"

    history = redis_client.get(key)

    if not history:
        return []

    return json.loads(history)


def save_conversation_history(
    conversation_id: str,
    history: list,
) -> None:
    logger.info("Function called: save conversation history")

    if not USE_REDIS:
        return

    key = f"conversation:{conversation_id}"

    redis_client.set(
        key,
        json.dumps(history),
        ex=MEMORY_TTL,
    )