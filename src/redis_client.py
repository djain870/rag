import os
from dotenv import load_dotenv

import redis

load_dotenv()
USE_REDIS = os.getenv("USE_REDIS", "true").lower() == "true"

redis_client = None

if USE_REDIS:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
    )