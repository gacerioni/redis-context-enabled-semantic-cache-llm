# src/memory/long_term.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import time, hashlib, math

from src.db.redis_client import r


# ---------- Model ----------

@dataclass
class LTMFact:
    id: str                  # stable id (hash of type+value)
    type: str                # e.g., "org", "role", "preference", "tool", "goal", "constraint"
    value: str               # the fact itself, normalized text
    source: str              # "manual", "conversation", "upload", etc.
    confidence: float        # 0..1
    first_seen: float        # epoch seconds
    last_seen: float         # epoch seconds
    count: int               # observation count
    expires_at: Optional[float] = None  # optional epoch seconds


# ---------- Keys ----------

def lt_map_key(user_id: str) -> str:
    # RedisJSON object: { "facts": { "<id>": <LTMFact>, ... }, "order": ["<id>", ...] }
    return f"user:{user_id}:ltm"

def legacy_array_key(user_id: str) -> str:
    # Old flat array location
    return f"user:{user_id}:longterm"


# ---------- Helpers ----------

def _fact_id(f_type: str, value: str) -> str:
    base = f"{f_type}::{value.strip().lower()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def _now() -> float:
    return time.time()

def _ensure_root(key: str):
    try:
        doc = r.json().get(key)
    except Exception:
        doc = None
    if not isinstance(doc, dict) or "facts" not in doc or "order" not in doc:
        r.json().set(key, "$", {"facts": {}, "order": []})

def _unwrap_path_get(val):
    """RedisJSON path GET may return a list (even length-1). Normalize to scalar/dict."""
    if isinstance(val, list):
        return val[0] if val else None
    return val

def _get_order_list(key: str) -> List[str]:
    try:
        order = r.json().get(key, "$.order")
        order = _unwrap_path_get(order)
        if isinstance(order, list):
            return list(order)
    except Exception:
        pass
    return []


# ---------- Legacy migration ----------

def migrate_legacy_array_if_present(user_id: str) -> int:
    """
    Migrate legacy flat list facts like:
      ["persona=rag_strict", "locale=pt-BR", "tone=concise", "random note"]
    into structured LTM facts. Returns the number of migrated items.
    """
    lkey = legacy_array_key(user_id)
    try:
        arr = r.json().get(lkey)
    except Exception:
        arr = None

    if not isinstance(arr, list) or not arr:
        return 0

    migrated = 0
    for item in arr:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item:
            continue

        # Support both "k=v" and bare strings
        if "=" in item:
            k, v = item.split("=", 1)
            f_type, value = k.strip(), v.strip()
        else:
            f_type, value = "note", item

        try:
            upsert_fact(
                user_id,
                f_type=f_type,
                value=value,
                source="legacy",
                confidence=0.6,
                ttl_seconds=None,
            )
            migrated += 1
        except Exception:
            # never crash the app on a bad legacy entry
            pass

    # Clear legacy after migration to avoid duplicate promotion
    try:
        r.json().set(lkey, "$", [])
    except Exception:
        pass

    return migrated


# ---------- Public API ----------

def upsert_fact(
    user_id: str,
    f_type: str,
    value: str,
    *,
    source: str = "conversation",
    confidence: float = 0.8,
    ttl_seconds: Optional[int] = None,
) -> LTMFact:
    """
    Insert/update a structured fact with dedupe + stats.
    """
    key = lt_map_key(user_id)
    _ensure_root(key)

    fid = _fact_id(f_type, value)
    path = f"$.facts.{fid}"
    now = _now()

    # read existing (path get can return a list)
    try:
        existing = r.json().get(key, path)
    except Exception:
        existing = None
    existing = _unwrap_path_get(existing)

    if isinstance(existing, dict) and existing:
        # update counters + recency (no overwrite of value/type)
        existing["last_seen"] = now
        existing["count"] = int(existing.get("count", 1)) + 1
        # keep max confidence, keep the most recent source (optional)
        existing["confidence"] = max(float(existing.get("confidence", 0.0)), float(confidence))
        existing["source"] = source or existing.get("source", "conversation")
        if ttl_seconds and ttl_seconds > 0:
            existing["expires_at"] = now + ttl_seconds
        try:
            r.json().set(key, path, existing)
        except Exception:
            # If a weird path error happens, recreate the root and set again
            _ensure_root(key)
            r.json().set(key, path, existing)
        fact = LTMFact(**existing)
    else:
        fact = LTMFact(
            id=fid,
            type=f_type,
            value=value.strip(),
            source=source,
            confidence=float(confidence),
            first_seen=now,
            last_seen=now,
            count=1,
            expires_at=(now + ttl_seconds) if (ttl_seconds and ttl_seconds > 0) else None,
        )
        r.json().set(key, path, asdict(fact))
        # push id to order (MRU at the end)
        try:
            r.json().arrappend(key, "$.order", fact.id)
        except Exception:
            # ensure structure then retry
            _ensure_root(key)
            r.json().arrappend(key, "$.order", fact.id)

    _prune_if_needed(user_id)
    return fact


def get_all_facts(user_id: str) -> List[LTMFact]:
    key = lt_map_key(user_id)
    try:
        doc = r.json().get(key) or {}
    except Exception:
        doc = {}
    facts_obj = (doc.get("facts") or {}) if isinstance(doc, dict) else {}
    out: List[LTMFact] = []
    now = _now()
    for obj in facts_obj.values():
        if not isinstance(obj, dict):
            continue
        # filter expired
        exp = obj.get("expires_at")
        if exp and now > float(exp):
            continue
        try:
            out.append(LTMFact(**obj))
        except Exception:
            # skip malformed records
            continue
    return out


def rank_facts(
    user_id: str,
    *,
    limit: int = 8,
    now: Optional[float] = None,
) -> List[LTMFact]:
    """
    Score = α*log(1+count) + β*recency_decay + γ*confidence
    recency_decay = exp(-(now - last_seen)/tau), tau ~ 14 days
    """
    facts = get_all_facts(user_id)
    if not facts:
        return []

    now = now or _now()
    tau = 14 * 24 * 3600.0  # ~2 weeks
    α, β, γ = 0.6, 0.3, 0.1

    def score(f: LTMFact) -> float:
        recency = math.exp(-(now - float(f.last_seen)) / tau)
        return α * math.log1p(max(0, int(f.count))) + β * recency + γ * float(f.confidence)

    ranked = sorted(facts, key=score, reverse=True)
    return ranked[:max(0, int(limit))]


def delete_fact(user_id: str, fid: str) -> bool:
    key = lt_map_key(user_id)
    _ensure_root(key)
    # remove object
    try:
        deleted = r.json().forget(key, f"$.facts.{fid}")  # returns number of paths removed
    except Exception:
        deleted = 0
    # remove from order
    order = _get_order_list(key)
    if fid in order:
        order.remove(fid)
        try:
            r.json().set(key, "$.order", order)
        except Exception:
            pass
    return bool(deleted)


def clear_all(user_id: str):
    key = lt_map_key(user_id)
    try:
        r.json().set(key, "$", {"facts": {}, "order": []})
    except Exception:
        pass


def _prune_if_needed(user_id: str, cap: int = 128):
    """
    Keep at most `cap` facts (drop the lowest scoring/oldest).
    """
    key = lt_map_key(user_id)
    order = _get_order_list(key)
    if len(order) <= cap:
        return
    # drop oldest from the front until size==cap
    to_drop = len(order) - cap
    for _ in range(to_drop):
        fid = order.pop(0)
        try:
            r.json().forget(key, f"$.facts.{fid}")
        except Exception:
            pass
    try:
        r.json().set(key, "$.order", order)
    except Exception:
        pass