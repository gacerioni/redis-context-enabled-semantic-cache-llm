# src/cache/semantic_cache.py
from __future__ import annotations

import json
import uuid
import time
import unicodedata
import hashlib
from typing import Any, Dict, Optional, List

from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from src.schema import cache_schema, CACHE_PREFIX
from src.llm import embed
from src.db.redis_client import r


# ---------------------------
# Helpers
# ---------------------------

def _normalize_prompt(s: str) -> str:
    """Lowercase, trim, and strip accents so similar phrasings hit the same cache entry."""
    if not s:
        return ""
    s = s.strip().lower()
    # Strip diacritics (e.g., "itália" -> "italia")
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return s


def ensure_cache_index() -> None:
    """Create the semantic cache index if it does not exist."""
    idx = SearchIndex.from_dict(cache_schema, client=r)
    if not idx.exists():
        idx.create(overwrite=False)


def _query_index(qvec: List[float], k: int) -> List[Dict[str, Any]]:
    """Run a vector query and normalize the return format to a list of rows."""
    idx = SearchIndex.from_dict(cache_schema, client=r)
    vq = VectorQuery(
        qvec,
        "embedding",  # vector field name
        return_fields=["prompt", "generic_answer", "meta"],
        num_results=k,
        return_score=True,
    )
    res = idx.query(vq)
    # redisvl can return a list or an object with `.results`
    rows = res if isinstance(res, list) else getattr(res, "results", [])
    return rows or []


def _load_meta(meta_obj: Any) -> Dict[str, Any]:
    """Meta may come back as a JSON string or dict; normalize to dict (best-effort)."""
    if isinstance(meta_obj, dict):
        return meta_obj
    if isinstance(meta_obj, str):
        try:
            return json.loads(meta_obj)
        except Exception:
            return {}
    return {}


# ---------------------------
# Public API
# ---------------------------

def lookup(
    prompt: str,
    k: int = 3,
    threshold: float = 0.22,
    context_signature: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find the closest cached generic answer for a (normalized) prompt.
    - Uses top-K and returns the best row whose distance <= threshold.
    - Distance is cosine distance; lower is better.
    - If `context_signature` is provided, require an exact match against the stored meta.
    """
    ensure_cache_index()

    norm = _normalize_prompt(prompt)
    qvec = embed([norm])[0]

    rows = _query_index(qvec, k=k)
    if not rows:
        return None

    score_field = VectorQuery.DISTANCE_ID
    # Pick closest under threshold
    best = None
    best_score = float("inf")
    for row in rows:
        try:
            score = float(row.get(score_field, 1.0))
        except Exception:
            continue
        if score < best_score:
            best = row
            best_score = score

    if best is None or best_score > float(threshold):
        return None

    # Guard by context signature when provided
    if context_signature:
        meta = _load_meta(best.get("meta", "{}"))
        if meta.get("context_signature") != context_signature:
            return None

    return {
        "prompt": best.get("prompt", ""),
        "generic_answer": best.get("generic_answer", ""),
        # keep meta as stored (string) for backward compatibility with callers
        "meta": best.get("meta", "{}"),
    }


def store(prompt: str, generic_answer: str, meta: Dict[str, Any]) -> None:
    """
    Store a generic answer keyed by the normalized prompt embedding.
    Adds a small meta block with created_at and a prompt hash for inspection.
    The caller may include `context_signature` in `meta` to bind cache entries
    to a specific identity/locale/persona context.
    """
    ensure_cache_index()

    norm = _normalize_prompt(prompt)
    emb = embed([norm])[0]
    key = f"{CACHE_PREFIX}{uuid.uuid4()}"
    qa_id = key.split(":")[-1]

    # Keep meta compact and JSON-serializable
    safe_meta = {
        **(meta or {}),
        "created_at": int(time.time()),
        "prompt_hash": hashlib.sha1(norm.encode()).hexdigest()[:10],
    }

    payload = {
        "qa_id": qa_id,
        "prompt": norm,
        "generic_answer": generic_answer,
        "meta": json.dumps(safe_meta),
        "embedding": emb,
    }
    r.json().set(key, "$", payload)