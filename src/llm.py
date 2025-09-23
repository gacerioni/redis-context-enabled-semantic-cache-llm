from typing import List, Dict
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBED_MODEL, PREMIUM_LLM, CHEAP_LLM

_oai = OpenAI(api_key=OPENAI_API_KEY)

def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def complete(system: str, messages: List[Dict[str, str]], model: str = PREMIUM_LLM,
             max_tokens: int = 500, temperature: float = 0.2) -> str:
    resp = _oai.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system}] + messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

PERSONALIZER_SYS = (
    "You are a helpful assistant that PERSONALIZES a generic answer using "
    "the provided user profile, recent chat messages, the semantic route, and RAG snippets. "
    "Respect the user's tone/locale from the profile. Keep responses concise, structured, and direct."
)

PREMIUM_SYS = (
    "You are a STRICT RAG assistant. Use ONLY the provided context blocks to answer: "
    "[USER PROFILE], [RECENT MESSAGES], [SEMANTIC ROUTE], and [RAG]. "
    "If the information is missing or insufficient, explicitly say what is unknown and do not invent facts. "
    "Prefer bullet points and short paragraphs. Keep answers concise and helpful."
)

CHEAP_MODEL = CHEAP_LLM
PREMIUM_MODEL = PREMIUM_LLM