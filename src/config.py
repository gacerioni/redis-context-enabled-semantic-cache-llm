import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
PREMIUM_LLM = os.getenv("PREMIUM_LLM", "gpt-4o")
CHEAP_LLM = os.getenv("CHEAP_LLM", "gpt-4o-mini")

SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))

# Chat memory
ST_HISTORY_TTL_SEC = 60 * 30
DEFAULT_USER_ID = "demo_user"