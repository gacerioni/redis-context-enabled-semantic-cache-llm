import json, uuid, time
from typing import Any, Dict, Optional
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from src.schema import cache_schema, CACHE_PREFIX
from src.llm import embed
from src.db.redis_client import r

def ensure_cache_index():
    idx = SearchIndex.from_dict(cache_schema, client=r)
    if not idx.exists():
        idx.create(overwrite=False)

def lookup(prompt: str, k: int = 1, threshold: float = 0.12) -> Optional[Dict[str, Any]]:
    ensure_cache_index()
    qvec = embed([prompt])[0]
    vq = VectorQuery(
        qvec,  # vector
        "embedding",  # vector_field_name
        return_fields=["prompt", "generic_answer", "meta"],
        num_results=k,
        return_score=True
    )
    idx = SearchIndex.from_dict(cache_schema, client=r)
    res = idx.query(vq)  # may be list or object

    rows = res if isinstance(res, list) else getattr(res, "results", [])
    if not rows:
        return None

    best = rows[0]
    score_field = VectorQuery.DISTANCE_ID
    if float(best[score_field]) <= threshold:
        return {
            "prompt": best["prompt"],
            "generic_answer": best["generic_answer"],
            "meta": best.get("meta", "{}"),
        }
    return None

def store(prompt: str, generic_answer: str, meta: Dict[str, Any]):
    ensure_cache_index()
    key = f"{CACHE_PREFIX}{uuid.uuid4()}"
    payload = {
        "qa_id": key.split(":")[-1],
        "prompt": prompt,
        "generic_answer": generic_answer,
        "meta": json.dumps({**meta, "created_at": int(time.time())}),
        "embedding": embed([prompt])[0],
    }
    r.json().set(key, "$", payload)