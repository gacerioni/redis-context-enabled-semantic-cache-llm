from typing import List, Dict, Any
from redisvl.index import SearchIndex
from src.schema import kb_schema, KB_PREFIX
from src.db.redis_client import r
from src.utils.chunker import simple_chunk
from src.llm import embed
import os  # NEW

class Doc:
    def __init__(self, doc_id: str, source: str, text: str):
        self.doc_id = doc_id
        self.source = source
        self.text = text

SEED_DOCS = [
    Doc("banking_faq_1", "kb/banking_faq_1",
        "Checking accounts allow unlimited withdrawals; savings may limit transfers. "
        "Interest varies by product. Wire transfers have cutoff times/fees; domestic wires can settle same-day."),
    Doc("trading_basics_1", "kb/trading_basics_1",
        "Market orders execute at best price. Limit orders set bounds. "
        "Stop-loss helps manage downside risk in volatility."),
    Doc("security_compliance_1", "kb/security_compliance_1",
        "Use least-privilege for customer data. Access reviews quarterly. MFA required for admin consoles."),
]

def ensure_kb_index():
    idx = SearchIndex.from_dict(kb_schema, client=r)
    if not idx.exists():
        idx.create(overwrite=False)

def upsert_kb(docs: List[Doc]):
    ensure_kb_index()
    pipe = r.pipeline()
    for d in docs:
        chunks = simple_chunk(d.text)
        embs = embed(chunks)
        file_name = os.path.basename(d.source) if d.source else ""  # NEW
        for i, (ch, vec) in enumerate(zip(chunks, embs)):
            key = f"{KB_PREFIX}{d.doc_id}:{i}"
            payload: Dict[str, Any] = {
                "doc_id": d.doc_id,
                "chunk_id": f"{d.doc_id}:{i}",
                "chunk_index": i,            # NEW
                "file_name": file_name,      # NEW
                "text": ch,
                "source": d.source,
                "embedding": vec,
            }
            pipe.json().set(key, "$", payload)
    pipe.execute()

def seed_if_empty():
    if not list(r.scan_iter(f"{KB_PREFIX}*")):
        upsert_kb(SEED_DOCS)