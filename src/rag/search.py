from typing import Any, Dict, List
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from src.schema import kb_schema
from src.llm import embed
from src.db.redis_client import r

def rag_search(query: str, topk: int = 3) -> List[Dict[str, Any]]:
    qvec = embed([query])[0]
    vq = VectorQuery(
        qvec,
        "embedding",
        return_fields=["text", "source", "doc_id", "chunk_id", "file_name", "chunk_index"],
        num_results=topk * 3,  # overfetch, then dedupe by doc
        return_score=True,
    )
    idx = SearchIndex.from_dict(kb_schema, client=r)
    res = idx.query(vq)

    rows = res if isinstance(res, list) else getattr(res, "results", [])
    score_field = VectorQuery.DISTANCE_ID  # e.g. "__v_score"

    # normalize + keep best chunk per doc_id
    best_by_doc: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        doc_id = row.get("doc_id", "")
        score = float(row.get(score_field, 0.0))
        item = {
            "text": row.get("text", ""),
            "source": row.get("source", ""),
            "doc_id": doc_id,
            "chunk_id": row.get("chunk_id", ""),
            "file_name": row.get("file_name", ""),
            "chunk_index": row.get("chunk_index", None),
            "score": score,
        }
        prev = best_by_doc.get(doc_id)
        if prev is None or score < prev["score"]:  # lower cosine distance = closer
            best_by_doc[doc_id] = item

    # sort by score asc (closest first) and cap to topk
    uniq = sorted(best_by_doc.values(), key=lambda x: x["score"])[:topk]
    return uniq