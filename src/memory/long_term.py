from typing import List
from src.db.redis_client import r

def lt_memory_key(user_id: str) -> str:
    return f"user:{user_id}:longterm"

def add_longterm_fact(user_id: str, fact: str):
    key = lt_memory_key(user_id)
    try:
        arr = r.json().get(key)
    except Exception:
        arr = None
    if not isinstance(arr, list):
        r.json().set(key, "$", [])
    r.json().arrappend(key, "$", fact)

def get_longterm(user_id: str) -> List[str]:
    key = lt_memory_key(user_id)
    try:
        arr = r.json().get(key)
    except Exception:
        arr = []
    return arr or []