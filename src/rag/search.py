from typing import Any, Dict, List
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from src.schema import kb_schema
from src.llm import embed
from src.db.redis_client import r

def rag_search(query: str, topk: int = 3) -> List[Dict[str, Any]]:
    qvec = embed([query])[0]
    vq = VectorQuery(
        qvec,                 # vector
        "embedding",          # vector_field_name
        return_fields=[
            "text",
            "source",
            "doc_id",
            "chunk_id",
            "file_name",       # NEW
            "chunk_index",     # NEW
        ],
        num_results=topk,
        return_score=True
    )
    idx = SearchIndex.from_dict(kb_schema, client=r)
    res = idx.query(vq)  # may return a list OR an object with .results

    # Normalize rows across redisvl variants
    rows = res if isinstance(res, list) else getattr(res, "results", [])

    score_field = VectorQuery.DISTANCE_ID  # e.g., "__v_score"
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append({
            "text": row.get("text", ""),
            "source": row.get("source", ""),
            "doc_id": row.get("doc_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "file_name": row.get("file_name", ""),              # NEW
            "chunk_index": row.get("chunk_index", None),        # NEW
            "score": float(row.get(score_field, 0.0)),
        })
    return out