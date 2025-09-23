import json, time
from typing import Dict, List
from src.db.redis_client import r
from src.config import ST_HISTORY_TTL_SEC

def st_history_key(session_id: str) -> str:
    return f"chat:{session_id}:history"

def append_short_term(session_id: str, role: str, content: str):
    key = st_history_key(session_id)
    entry = json.dumps({"t": int(time.time()), "role": role, "content": content})
    r.rpush(key, entry); r.expire(key, ST_HISTORY_TTL_SEC)

def get_short_term(session_id: str, k: int = 6) -> List[Dict[str, str]]:
    key = st_history_key(session_id)
    n = r.llen(key)
    if n <= 0:
        return []
    raw = r.lrange(key, max(0, n - k), -1)
    return [json.loads(x) for x in raw]