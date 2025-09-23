from typing import Dict
from src.db.redis_client import r

def profile_key(user_id: str) -> str:
    return f"user:{user_id}:profile"

def upsert_profile(user_id: str, profile: Dict[str, str]):
    r.hset(profile_key(user_id), mapping=profile)

def get_profile(user_id: str) -> Dict[str, str]:
    return r.hgetall(profile_key(user_id)) or {}