# src/workflows.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Optional

from src.llm import (
    complete,
    PERSONALIZER_SYS as DEFAULT_PERSONALIZER_SYS,
    PREMIUM_SYS as DEFAULT_PREMIUM_SYS,
    CHEAP_MODEL,
    PREMIUM_MODEL,
)

# ---- LTM imports (structured API preferred, fallback to legacy) ----
_LTM_STRUCTURED = False
try:
    from src.memory.long_term import (
        upsert_fact,
        rank_facts,
        get_all_facts,
        delete_fact,
        LTMFact,
    )  # type: ignore
    _LTM_STRUCTURED = True
except Exception:
    from src.memory.long_term import get_longterm  # type: ignore

from src.rag.search import rag_search
from src.cache import semantic_cache as sc
from src.profiles.user_profile import get_profile
from src.memory.short_term import get_short_term
from src.routing.semantic_router import route_query


# ---------------------------
# Persona-aware system prompts
# ---------------------------

BASE_PERSONALIZER = DEFAULT_PERSONALIZER_SYS
BASE_PREMIUM = DEFAULT_PREMIUM_SYS

PERSONA_PROMPTS: Dict[str, Dict[str, str]] = {
    "rag_strict": {
        "personalizer": (
            "You are a helpful assistant that PERSONALIZES a generic answer using the provided "
            "user profile, long-term facts, recent chat messages, the semantic route, and RAG snippets. "
            "Respect the user's tone/locale. Keep responses concise and structured. "
            "If context is insufficient, clearly state what is missing."
        ),
        "premium": (
            "You are a STRICT RAG assistant. Use ONLY the provided context blocks to answer: "
            "[USER PROFILE], [LONG-TERM FACTS], [RECENT MESSAGES], [SEMANTIC ROUTE], and [RAG]. "
            "If the information is missing or insufficient, say so explicitly and do not invent facts. "
            "Prefer bullet points and short paragraphs."
        ),
    },
    "creative_helper": {
        "personalizer": (
            "You are a creative, friendly assistant. Personalize the generic answer using the user's profile, long-term facts, "
            "recent chat, semantic route and RAG snippets. Be engaging but precise; do not invent facts. "
            "Offer 1–2 extra helpful suggestions when appropriate."
        ),
        "premium": (
            "You are a creative RAG assistant. Use the provided context blocks to answer clearly. "
            "If context lacks details, you may add general best-practice guidance, but mark it as general advice. "
            "Keep a friendly, helpful tone and avoid making up specific facts."
        ),
    },
    "analyst": {
        "personalizer": (
            "You are an analytical assistant. Personalize the generic answer using profile, long-term facts, recent chat, route, and RAG. "
            "Be structured: list assumptions, steps, and trade-offs. Keep it concise and evidence-based."
        ),
        "premium": (
            "You are an analytical RAG assistant. Use ONLY the provided context blocks. "
            "Present the answer with numbered steps, key assumptions, and risks. "
            "If context is insufficient, list the missing data and propose how to obtain it."
        ),
    },
    "support_agent": {
        "personalizer": (
            "You are an empathetic support agent. Personalize the generic answer using profile, long-term facts, recent chat, route, and RAG. "
            "Acknowledge the user's situation, then give clear step-by-step guidance. Avoid jargon."
        ),
        "premium": (
            "You are a support-focused RAG assistant. Use ONLY the provided context blocks. "
            "Start with a brief acknowledgment, then provide step-by-step instructions. "
            "If context is missing, state what is needed next and possible next actions."
        ),
    },
}

PERSONA_TEMPS: Dict[str, Dict[str, float]] = {
    "rag_strict": {"personalizer": 0.2, "premium": 0.1},
    "creative_helper": {"personalizer": 0.5, "premium": 0.4},
    "analyst": {"personalizer": 0.25, "premium": 0.2},
    "support_agent": {"personalizer": 0.25, "premium": 0.2},
}


def _select_prompts_and_temps(persona: str) -> Tuple[str, str, float, float]:
    key = (persona or "").lower().strip()
    prompts = PERSONA_PROMPTS.get(key)
    temps = PERSONA_TEMPS.get(key)
    personalizer_sys = prompts["personalizer"] if prompts else BASE_PERSONALIZER
    premium_sys = prompts["premium"] if prompts else BASE_PREMIUM
    t_personalizer = temps["personalizer"] if temps else 0.2
    t_premium = temps["premium"] if temps else 0.2
    return personalizer_sys, premium_sys, t_personalizer, t_premium


# ---------------------------
# Fact extraction
# ---------------------------

# Tipos com valor único (mantemos "uma verdade")
_SINGLETON_TYPES = {
    "name", "company",
    "location", "location_city", "location_country",
    "timezone", "language", "locale",
    "role", "org", "team",
    "persona", "tone",
    "contact_preference", "currency",
    "risk_profile", "investment_horizon",
}

_PREF_PAT = re.compile(r"\b(prefiro|prefer|gosto de|i prefer)\s+([^.;,\n]+)", re.IGNORECASE)
_WORK_PAT = re.compile(r"\b(trabalho (?:no|na|em)|work at|i work at)\s+([^.;,\n]+)", re.IGNORECASE)
_TEAM_PAT = re.compile(r"\b(?:time|squad|equipe|team)\s+([^.;,\n]+)", re.IGNORECASE)

# Localização (frases comuns PT/EN)
_LOC_PATS = [
    re.compile(r"\b(?:moro|vivo|resido)\s+(?:no|na|em)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\b(?:sou\s+de|sou\s+do|sou\s+da)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\b(?:based in|i live in|i'm from|im from)\s+([^.;,\n]+)", re.IGNORECASE),
]

# Idioma/locale
_LANG_PATS = [
    re.compile(r"\b(?:falo|prefiro)\s+(portugu[eê]s|ingl[eê]s)\b", re.IGNORECASE),
    re.compile(r"\b(?:language)\s*[:\-]?\s*(english|portuguese)\b", re.IGNORECASE),
]
_LANG_MAP = {
    "português": "pt-BR", "portugues": "pt-BR",
    "english": "en-US", "inglês": "en-US", "ingles": "en-US",
}

# Fuso horário
_TZ_PATS = [
    re.compile(r"\b(?:timezone|fuso\s*hor[aá]rio)\s*[:\-]?\s*([a-zA-Z_\/\-+0-9]+)", re.IGNORECASE),
    re.compile(r"\bUTC ?([+\-]\d{1,2})\b", re.IGNORECASE),
]

# Ferramentas
_TOOLS_PATS = [
    re.compile(r"\b(?:uso|usamos|trabalh(?:o|amos)\s+com)\s+(kubernetes|docker|redis|postgres|mysql|vs\s*code|pycharm)\b", re.IGNORECASE),
    re.compile(r"\b(?:we\s+use|using)\s+(kubernetes|docker|redis|postgres|mysql|vs\s*code|pycharm)\b", re.IGNORECASE),
]

# Expertise / metas / restrições
_EXPERT_PATS = [
    re.compile(r"\b(?:sou especialista|tenho experi[eê]ncia em)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\b(?:experienced in|expert in)\s+([^.;,\n]+)", re.IGNORECASE),
]
_GOAL_PATS = [
    re.compile(r"\b(?:quero|pretendo|planejo)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\b(?:i want to|i plan to)\s+([^.;,\n]+)", re.IGNORECASE),
]
_CONSTR_PATS = [
    re.compile(r"\b(?:n[aã]o posso|tenho NDA|sem acesso a)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\b(?:cannot|under NDA|no access to)\s+([^.;,\n]+)", re.IGNORECASE),
]

# Nome / empresa
_NAME_PATS = [
    re.compile(r"\bme chamo\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\bmeu nome (?:é|eh)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\bI(?:'| a)m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
]
_COMPANY_PATS = [
    re.compile(r"\btrabalho (?:no|na|em)\s+([^.;,\n]+)", re.IGNORECASE),
    re.compile(r"\bwork (?:at|for)\s+([^.;,\n]+)", re.IGNORECASE),
]

# Cidade/país (além de _LOC_PATS)
_CITY_PATS = [re.compile(r"\b(?:moro|estou)\s+em\s+([^.;,\n]+)", re.IGNORECASE)]
_COUNTRY_PATS = [re.compile(r"\b(?:no|na)\s+(brasil|brazil|eua|estados unidos|usa|portugal|spain)\b", re.IGNORECASE)]

# Contato / moeda
_CONTACT_PATS = [
    re.compile(r"\b(?:prefiro|fale comigo) (?:por|via)\s+(e-?mail|email|whatsapp|telefone|call)", re.IGNORECASE),
    re.compile(r"\bprefer (?:to\s+be\s+contacted\s+)?via\s+(email|whatsapp|phone|call)", re.IGNORECASE),
]
_CURRENCY_PATS = [
    re.compile(r"\b(?:trabalhar|cotar|orçar)\s+em\s+(brl|usd|eur|real|dólar|dolar|euro)", re.IGNORECASE),
    re.compile(r"\b(?:currency|moeda)\s*[:\-]?\s*(brl|usd|eur)", re.IGNORECASE),
]

# Perfil de risco / horizonte
_RISK_PATS = [
    re.compile(r"\bperfil de risco\s*[:\-]?\s*(conservador|moderado|arrojado)", re.IGNORECASE),
    re.compile(r"\b(risk profile)\s*[:\-]?\s*(conservative|moderate|aggressive)", re.IGNORECASE),
]
_HORIZON_PATS = [
    re.compile(r"\b(?:horizonte|prazo)\s*(?:de|:)?\s*(curto|m[eé]dio|longo)", re.IGNORECASE),
    re.compile(r"\b(?:investment horizon)\s*[:\-]?\s*(short|medium|long)", re.IGNORECASE),
]

# Correções / mudanças (gatilho para sobrescrever singletons)
_CORRECTION_PATS = [re.compile(r"\b(na verdade|corrigindo|mudei|me mudei|agora)\b", re.IGNORECASE)]


def _norm_place(s: str) -> str:
    s = s.strip().lower()
    if s in {"brasil", "brazil"}:
        return "Brazil"
    if s in {"sp", "sao paulo", "são paulo"}:
        return "São Paulo"
    return s.title()


def _norm_currency(s: str) -> str:
    m = s.strip().lower()
    return {
        "brl": "BRL", "real": "BRL",
        "usd": "USD", "dólar": "USD", "dolar": "USD",
        "eur": "EUR", "euro": "EUR",
    }.get(m, s.upper())


def _extract_candidate_facts(msg: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not msg:
        return out

    # preferências / org / time
    if m := _PREF_PAT.search(msg):
        out.append({"type": "preference", "value": m.group(2).strip()})
    if m := _WORK_PAT.search(msg):
        out.append({"type": "org", "value": m.group(2).strip()})
    if m := _TEAM_PAT.search(msg):
        out.append({"type": "team", "value": m.group(1).strip()})

    # localização (genérica)
    for pat in _LOC_PATS:
        if m := pat.search(msg):
            out.append({"type": "location", "value": _norm_place(m.group(1))})
            break

    # idioma/locale
    for pat in _LANG_PATS:
        if m := pat.search(msg):
            val = _LANG_MAP.get(m.group(1).lower(), m.group(1))
            out.append({"type": "locale", "value": val})
            break

    # timezone
    for pat in _TZ_PATS:
        if m := pat.search(msg):
            tz = m.group(1).upper()
            if tz.startswith(("+", "-")):
                tz = f"UTC{tz}"
            out.append({"type": "timezone", "value": tz})
            break

    # ferramentas / expertise / metas / restrições
    for pat in _TOOLS_PATS:
        if m := pat.search(msg):
            out.append({"type": "tool", "value": m.group(1)})
            break
    for pat in _EXPERT_PATS:
        if m := pat.search(msg):
            out.append({"type": "expertise", "value": m.group(1).strip()})
            break
    for pat in _GOAL_PATS:
        if m := pat.search(msg):
            out.append({"type": "goal", "value": m.group(1).strip()})
            break
    for pat in _CONSTR_PATS:
        if m := pat.search(msg):
            out.append({"type": "constraint", "value": m.group(1).strip()})
            break

    # nome
    for pat in _NAME_PATS:
        if m := pat.search(msg):
            out.append({"type": "name", "value": m.group(1).strip()})
            break

    # empresa (se quiser manter separado de "org")
    for pat in _COMPANY_PATS:
        if m := pat.search(msg):
            out.append({"type": "company", "value": m.group(1).strip()})
            break

    # cidade / país (opcionais além de location)
    for pat in _CITY_PATS:
        if m := pat.search(msg):
            out.append({"type": "location_city", "value": _norm_place(m.group(1))})
            break
    for pat in _COUNTRY_PATS:
        if m := pat.search(msg):
            out.append({"type": "location_country", "value": _norm_place(m.group(1))})
            break

    # contato / moeda
    for pat in _CONTACT_PATS:
        if m := pat.search(msg):
            out.append({"type": "contact_preference", "value": m.group(1).lower()})
            break
    for pat in _CURRENCY_PATS:
        if m := pat.search(msg):
            out.append({"type": "currency", "value": _norm_currency(m.group(1))})
            break

    # risco / horizonte
    for pat in _RISK_PATS:
        if m := pat.search(msg):
            out.append({"type": "risk_profile", "value": (m.group(2) if m.lastindex == 2 else m.group(1)).lower()})
            break
    for pat in _HORIZON_PATS:
        if m := pat.search(msg):
            out.append({"type": "investment_horizon", "value": m.group(1).lower()})
            break

    # flag de correção
    if any(p.search(msg) for p in _CORRECTION_PATS):
        out.append({"type": "_correction_flag", "value": "true"})

    return out


def _upsert_unique(user_id: str, ftype: str, value: str, *, source: str, confidence: float):
    """
    Garante valor único para tipos singleton:
    remove entradas antigas do mesmo tipo e grava a nova.
    """
    try:
        existing = get_all_facts(user_id)
        for f in existing:
            if f.type == ftype and f.value.lower() != value.lower():
                delete_fact(user_id, f.id)
    except Exception:
        pass
    upsert_fact(user_id, f_type=ftype, value=value, source=source, confidence=confidence)


def _promote_facts_from_turn(user_id: str, message: str) -> None:
    if not _LTM_STRUCTURED or not message:
        return
    facts = _extract_candidate_facts(message)
    correction = any(f["type"] == "_correction_flag" for f in facts)
    for f in facts:
        if f["type"] == "_correction_flag":
            continue
        try:
            if f["type"] in _SINGLETON_TYPES:
                _upsert_unique(
                    user_id,
                    f["type"],
                    f["value"],
                    source="conversation",
                    confidence=0.8 if correction else 0.75,
                )
            else:
                upsert_fact(
                    user_id,
                    f_type=f["type"],
                    value=f["value"],
                    source="conversation",
                    confidence=0.7,
                )
        except Exception:
            # nunca quebre o fluxo se LTM falhar
            pass


# ---------------------------
# Profile resolution & cache safety
# ---------------------------

def _resolve_singletons(user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overlay LTM singletons on top of the profile hash.
    Ensures we don't contradict updated 'name', 'location', etc.
    """
    if not _LTM_STRUCTURED:
        return dict(profile or {})
    resolved = dict(profile or {})
    try:
        for f in rank_facts(user_id, limit=64) or []:
            if f.type in _SINGLETON_TYPES:
                resolved[f.type] = f.value
    except Exception:
        pass
    return resolved


_SENSITIVE_Q_PATTERNS = [
    re.compile(r"\b(my|meu|minha)\s+name\b", re.IGNORECASE),
    re.compile(r"\bqual\s+e?\s*o?\s*meu\s+nome\b", re.IGNORECASE),
    re.compile(r"\bonde\s+eu\s+moro\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+do\s+i\s+live\b", re.IGNORECASE),
    re.compile(r"\b(mudei|me mudei|agora|na verdade|corrigindo)\b", re.IGNORECASE),
    re.compile(r"\b(locale|idioma|language|timezone|fuso)\b", re.IGNORECASE),
]

def _is_sensitive_query(q: str) -> bool:
    return bool(q and any(p.search(q) for p in _SENSITIVE_Q_PATTERNS))


# ---------------------------
# Context building
# ---------------------------

def _ltm_block(user_id: str) -> str:
    """
    Monta o bloco textual de LTM. Se estruturado, prioriza singletons e
    completa com top-ranked facts. No legado, imprime a lista plana.
    """
    try:
        if _LTM_STRUCTURED:
            facts = rank_facts(user_id, limit=16) or []
            if not facts:
                return ""

            # Promova singletons para o topo (uma visão mais estável no contexto)
            singletons = [f for f in facts if f.type in _SINGLETON_TYPES]
            others = [f for f in facts if f.type not in _SINGLETON_TYPES]
            ordered = singletons + others
            # dedupe mantendo a primeira ocorrência
            seen_ids = set()
            final: List[LTMFact] = []
            for f in ordered:
                if f.id in seen_ids:
                    continue
                seen_ids.add(f.id)
                final.append(f)

            lines = [f"- [{f.type}] {f.value} (seen {getattr(f, 'count', 1)}×)" for f in final[:8]]
            return "[LONG-TERM FACTS]\n" + "\n".join(lines)

        else:
            arr = get_longterm(user_id)
            if not arr:
                return ""
            return "[LONG-TERM FACTS]\n" + "\n".join(f"- {s}" for s in arr)

    except Exception:
        return ""


def build_context_blocks(user_id: str, session_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
    raw_profile = get_profile(user_id) or {}
    profile = _resolve_singletons(user_id, raw_profile)  # <-- LTM overrides
    st = get_short_term(session_id) or []
    route = route_query(query) or {}
    rag = rag_search(query) or []

    ctx: Dict[str, Any] = {"profile": profile, "recent": st, "rag": rag, "route": route}

    parts: List[str] = []
    if profile:
        parts.append("[USER PROFILE]\n" + json.dumps(profile, ensure_ascii=False))
    if (ltm_text := _ltm_block(user_id)):
        parts.append(ltm_text)
    if st:
        parts.append("[RECENT MESSAGES]\n" + "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in st))
    if route.get("name"):
        parts.append(f"[SEMANTIC ROUTE]\nname: {route['name']}  distance: {route.get('distance', 0.0):.4f}")
    if rag:
        parts.append("[RAG]\n" + "\n".join(
            f"• {x.get('text','')} (src: {x.get('file_name') or x.get('source','')}, score: {x.get('score',0.0):.4f})"
            for x in rag
        ))

    return "\n\n".join(parts), ctx


# ---------------------------
# Main orchestration
# ---------------------------

def answer_one(user_id: str, session_id: str, user_query: str) -> str:
    # 0) Promote candidate facts from this turn into LTM
    _promote_facts_from_turn(user_id, user_query)

    # 1) Load profile and persona
    raw_profile = get_profile(user_id) or {}
    resolved_profile = _resolve_singletons(user_id, raw_profile)  # overlay singletons from LTM
    persona = (raw_profile.get("persona") or raw_profile.get("mode") or "rag_strict").lower().strip()

    # 2) Select system prompts and temperatures
    personalizer_sys, premium_sys, t_personalizer, t_premium = _select_prompts_and_temps(persona)

    # 3) Build full context
    context_text, ctx = build_context_blocks(user_id, session_id, user_query)
    route_name = ctx.get("route", {}).get("name")
    route_tag = f"[route: {route_name}]" if route_name else "[route: none]"
    persona_tag = f"[persona: {persona}]"

    # ---- Context signature: binds cache entries to stable profile keys
    sig_fields = {
        "persona": persona,
        "locale": resolved_profile.get("locale"),
        "name": resolved_profile.get("name"),
    }
    sig_str = json.dumps(sig_fields, sort_keys=True, ensure_ascii=False)
    import hashlib
    context_signature = hashlib.sha1(sig_str.encode("utf-8")).hexdigest()[:12]

    # 4) Semantic cache lookup (skip for sensitive queries)
    hit = None if _is_sensitive_query(user_query) else sc.lookup(
        user_query,
        context_signature=context_signature
    )

    # 5) HIT → personalize cached generic answer
    if hit:
        generic = hit.get("generic_answer", "")
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

    # 6) MISS → run premium model with RAG and store generic answer
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
            "route": route_name,
            "persona": persona,
            "context_signature": context_signature,
        },
    )
    return f"{answer}\n\n_(Cache miss → RAG + premium model; stored generic answer)_\n{route_tag} {persona_tag}"