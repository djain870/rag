from src.logger import logger

from .redis_client import redis_client

RATE_LIMIT = 10
WINDOW_SECONDS = 60


def check_rate_limit(client_id: str) -> bool:
    logger.info("Function called: check rate limit")

    key = f"rate_limit:{client_id}"

    current_count = redis_client.incr(key)

    if current_count == 1:
        redis_client.expire(key, WINDOW_SECONDS)

    return current_count <= RATE_LIMIT