import json

from src.redis_client import redis_client


CACHE_TTL = 3600


def get_cached_response(
    conversation_id: str,
    query: str,
):
    key = f"cache:{conversation_id}:{query}"

    cached = redis_client.get(key)

    if not cached:
        return None

    return json.loads(cached)


def save_cached_response(
    conversation_id: str,
    query: str,
    response: dict,
) -> None:
    key = f"cache:{conversation_id}:{query}"

    redis_client.set(
        key,
        json.dumps(response),
        ex=CACHE_TTL,
    )