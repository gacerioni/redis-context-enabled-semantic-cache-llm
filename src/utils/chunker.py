from typing import List
from src.utils.tokens import count_tokens
import re

def simple_chunk(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur, cur_tok = [], [], 0
    for s in sents:
        t = count_tokens(s)
        if cur_tok + t > max_tokens and cur:
            chunks.append(" ".join(cur))
            back, ot = [], 0
            for sent in reversed(cur):
                tt = count_tokens(sent)
                if ot + tt <= overlap:
                    back.append(sent); ot += tt
                else:
                    break
            cur = list(reversed(back)); cur_tok = sum(count_tokens(x) for x in cur)
        cur.append(s); cur_tok += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks