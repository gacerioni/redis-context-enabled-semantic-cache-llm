import json
from typing import Any, Dict, Tuple

from src.llm import complete, PERSONALIZER_SYS as DEFAULT_PERSONALIZER_SYS, PREMIUM_SYS as DEFAULT_PREMIUM_SYS
from src.llm import CHEAP_MODEL, PREMIUM_MODEL
from src.memory.long_term import get_longterm
from src.rag.search import rag_search
from src.cache import semantic_cache as sc
from src.profiles.user_profile import get_profile
from src.memory.short_term import get_short_term
from src.routing.semantic_router import route_query


# ---------------------------
# Persona-aware system prompts
# ---------------------------

# Baselines come from src/llm.py; we use them as fallbacks for unknown personas.
BASE_PERSONALIZER = DEFAULT_PERSONALIZER_SYS
BASE_PREMIUM = DEFAULT_PREMIUM_SYS

PERSONA_PROMPTS: Dict[str, Dict[str, str]] = {
    # Strict RAG: never invent; always cite/use provided context blocks.
    "rag_strict": {
        "personalizer": (
            "You are a helpful assistant that PERSONALIZES a generic answer using the provided "
            "user profile, recent chat messages, the semantic route, and RAG snippets. "
            "Respect the user's tone/locale. Keep responses concise and structured. "
            "If context is insufficient, clearly state what is missing."
        ),
        "premium": (
            "You are a STRICT RAG assistant. Use ONLY the provided context blocks to answer: "
            "[USER PROFILE], [LONG-TERM MEMORY], [RECENT MESSAGES], [SEMANTIC ROUTE], and [RAG]. "
            "If the information is missing or insufficient, say so explicitly and do not invent facts. "
            "Prefer bullet points and short paragraphs."
        ),
    },

    # Creative helper: allowed to generalize and rephrase, but still grounded when context exists.
    "creative_helper": {
        "personalizer": (
            "You are a creative, friendly assistant. Personalize the generic answer using the user's profile, recent chat, "
            "semantic route and RAG snippets. Be engaging but precise; do not invent facts. "
            "Offer 1–2 extra helpful suggestions when appropriate."
        ),
        "premium": (
            "You are a creative RAG assistant. Use the provided context blocks to answer clearly. "
            "If context lacks details, you may add general best-practice guidance, but mark it as general advice. "
            "Keep a friendly, helpful tone and avoid making up specific facts."
        ),
    },

    # Analyst: structured reasoning, explicit steps, assumptions, trade-offs.
    "analyst": {
        "personalizer": (
            "You are an analytical assistant. Personalize the generic answer using profile, recent chat, route, and RAG. "
            "Be structured: list assumptions, steps, and trade-offs. Keep it concise and evidence-based."
        ),
        "premium": (
            "You are an analytical RAG assistant. Use ONLY the provided context blocks. "
            "Present the answer with numbered steps, key assumptions, and risks. "
            "If context is insufficient, list the missing data and propose how to obtain it."
        ),
    },

    # Support agent: empathetic, procedural, troubleshooting.
    "support_agent": {
        "personalizer": (
            "You are an empathetic support agent. Personalize the generic answer using profile, recent chat, route, and RAG. "
            "Acknowledge the user's situation, then give clear step-by-step guidance. Avoid jargon."
        ),
        "premium": (
            "You are a support-focused RAG assistant. Use ONLY the provided context blocks. "
            "Start with a brief acknowledgment, then provide step-by-step instructions. "
            "If context is missing, state what is needed next and possible next actions."
        ),
    },
}

# Temperature / verbosity knobs per persona (tweak as you like)
PERSONA_TEMPS: Dict[str, Dict[str, float]] = {
    "rag_strict":      {"personalizer": 0.2, "premium": 0.1},
    "creative_helper": {"personalizer": 0.5, "premium": 0.4},
    "analyst":         {"personalizer": 0.25, "premium": 0.2},
    "support_agent":   {"personalizer": 0.25, "premium": 0.2},
}


def _select_prompts_and_temps(persona: str) -> Tuple[str, str, float, float]:
    """Return (personalizer_sys, premium_sys, temp_personalizer, temp_premium) for a persona."""
    key = (persona or "").lower().strip()
    prompts = PERSONA_PROMPTS.get(key)
    temps = PERSONA_TEMPS.get(key)

    # Fallback to baselines if persona not recognized
    personalizer_sys = prompts["personalizer"] if prompts else BASE_PERSONALIZER
    premium_sys = prompts["premium"] if prompts else BASE_PREMIUM
    t_personalizer = temps["personalizer"] if temps else 0.2
    t_premium = temps["premium"] if temps else 0.2
    return personalizer_sys, premium_sys, t_personalizer, t_premium


# ---------------------------
# Context building
# ---------------------------

def build_context_blocks(user_id: str, session_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
    profile = get_profile(user_id)
    st = get_short_term(session_id)
    ltm = get_longterm(user_id)
    route = route_query(query)
    rag = rag_search(query)

    ctx: Dict[str, Any] = {
        "profile": profile,
        "longterm": ltm,
        "recent": st,
        "rag": rag,
        "route": route,
    }

    parts = []
    if profile:
        parts.append("[USER PROFILE]\n" + json.dumps(profile, ensure_ascii=False))
    if ltm:
        parts.append("[LONG-TERM MEMORY]\n" + "\n".join(ltm))
    if st:
        short = "\n".join(f"{m['role']}: {m['content']}" for m in st)
        parts.append("[RECENT MESSAGES]\n" + short)
    if route.get("name"):
        parts.append(f"[SEMANTIC ROUTE]\nname: {route['name']}  distance: {route['distance']:.4f}")
    if rag:
        snips = "\n".join(f"• {x['text']} (src: {x['source']})" for x in rag)
        parts.append("[RAG]\n" + snips)

    return "\n\n".join(parts), ctx


# ---------------------------
# Main orchestration
# ---------------------------

def answer_one(user_id: str, session_id: str, user_query: str) -> str:
    # Load persona/mode from profile (set in UI)
    profile = get_profile(user_id) or {}
    persona = (profile.get("persona") or profile.get("mode") or "rag_strict").lower().strip()

    # Select prompts and temperatures for this persona
    personalizer_sys, premium_sys, t_personalizer, t_premium = _select_prompts_and_temps(persona)

    # Build full context (profile + LTM + STM + route + RAG)
    hit = sc.lookup(user_query)
    context_text, ctx = build_context_blocks(user_id, session_id, user_query)
    route_tag = f"[route: {ctx['route']['name']}]" if ctx.get("route", {}).get("name") else "[route: none]"
    persona_tag = f"[persona: {persona}]"

    # ---- Semantic cache HIT → personalize with cheap model ----
    if hit:
        generic = hit["generic_answer"]
        msg = [
            {"role": "user", "content": f"Generic answer to personalize:\n{generic}"},
            {"role": "user", "content": f"Additional context for personalization:\n{context_text}"},
            {"role": "user", "content": f"User query (for final tweaks): {user_query}"},
        ]
        out = complete(
            system=personalizer_sys,
            messages=msg,
            model=CHEAP_MODEL,
            max_tokens=500,
            temperature=t_personalizer,
        )
        return f"{out}\n\n_(Semantic cache hit → personalized)_\n{route_tag} {persona_tag}"

    # ---- Semantic cache MISS → RAG + premium model, then store generic answer ----
    msg = [
        {"role": "user", "content": f"Question: {user_query}"},
        {"role": "user", "content": f"Context to use strictly:\n{context_text}"},
    ]
    answer = complete(
        system=premium_sys,
        messages=msg,
        model=PREMIUM_MODEL,
        max_tokens=600,
        temperature=t_premium,
    )

    sc.store(
        user_query,
        answer,
        {
            "kb_used": True,
            "route": ctx.get("route", {}).get("name"),
            "persona": persona,
        },
    )
    return f"{answer}\n\n_(Cache miss → RAG + premium model; stored generic answer)_\n{route_tag} {persona_tag}"