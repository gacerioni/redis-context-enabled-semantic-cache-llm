# src/routing/semantic_router.py
import os
from typing import Dict, Any
from redisvl.extensions.router import SemanticRouter, Route
from redisvl.utils.vectorize import OpenAITextVectorizer
from src.config import REDIS_URL, EMBED_MODEL

# Dica: deixe o threshold ~0.70–0.78. Mais baixo = mais “fácil” de casar rota.
# Adicione exemplos em pt-BR e en-US para robustez.

technology = Route(
    name="technology",
    references=[
        "latest advancements in AI",
        "newest gadgets",
        "what's trending in tech",
        "quantum computing news",
        "is 5G available everywhere",
        "explain edge computing",
        "tendências em tecnologia",
        "o que é computação de borda",
    ],
    metadata={"category": "tech"},
    distance_threshold=0.72,
)

sports = Route(
    name="sports",
    references=[
        "who won the game last night",
        "upcoming sports events",
        "latest sports news",
        "results for NBA and NFL",
        "cricket match updates",
        "Olympics schedule",
        "jogo do corinthians",
        "tabela do brasileirão",
    ],
    metadata={"category": "sports"},
    distance_threshold=0.72,
)

entertainment = Route(
    name="entertainment",
    references=[
        "top movies right now",
        "who won the Oscars",
        "celebrity news",
        "upcoming TV shows and films",
        "trending series on Netflix",
        "novidades no entretenimento",
    ],
    metadata={"category": "entertainment"},
    distance_threshold=0.70,
)

finance = Route(
    name="finance",
    references=[
        "latest stock market trends",
        "bitcoin price update",
        "how to invest in ETFs",
        "interest rate changes",
        "best budgeting tips",
        "explain inflation",
        "como investir em renda fixa",
        "taxa selic",
        "CDI vs poupança",
    ],
    metadata={"category": "finance"},
    distance_threshold=0.73,
)

health = Route(
    name="health",
    references=[
        "tips for mental health",
        "how to lose weight safely",
        "flu and covid symptoms",
        "healthy diets and routines",
        "benefits of meditation",
        "latest health research",
        "alimentação saudável",
        "sintomas de gripe",
    ],
    metadata={"category": "health"},
    distance_threshold=0.74,
)

travel = Route(
    name="travel",
    references=[
        "top destinations for 2025",
        "is Japan open for travel",
        "budget travel tips",
        "visa requirements for US",
        "backpacking Europe",
        "travel safety advice",
        "dicas de viagem baratas",
        "preciso de visto para os EUA",
    ],
    metadata={"category": "travel"},
    distance_threshold=0.72,
)

education = Route(
    name="education",
    references=[
        "best online learning platforms",
        "AI in classrooms",
        "how to learn coding",
        "top universities in Europe",
        "study tips for students",
        "education trends",
        "plataformas de estudo online",
        "como aprender programação",
    ],
    metadata={"category": "education"},
    distance_threshold=0.73,
)

food = Route(
    name="food",
    references=[
        "best recipes for dinner",
        "easy vegan meals",
        "restaurants near me",
        "what's trending in food",
        "how to cook steak properly",
        "healthy snack ideas",
        "restaurantes perto de mim",
        "receitas fáceis",
    ],
    metadata={"category": "food"},
    distance_threshold=0.71,
)

# ---- Novas rotas genéricas úteis para demos corporativas ----

coding = Route(
    name="coding",
    references=[
        "how to write python code",
        "debug this error",
        "explain this algorithm",
        "help with unit tests",
        "melhor prática em API design",
        "escreva uma função em javascript",
    ],
    metadata={"category": "dev"},
    distance_threshold=0.73,
)

devops = Route(
    name="devops",
    references=[
        "docker compose not starting",
        "kubernetes pod failing",
        "ci/cd pipeline tips",
        "observability best practices",
        "infra as code examples",
        "helm chart issues",
    ],
    metadata={"category": "devops"},
    distance_threshold=0.74,
)

documentation = Route(
    name="documentation",
    references=[
        "summarize this doc",
        "extract key points from PDF",
        "create a knowledge base article",
        "improve clarity of this README",
        "documentação do projeto",
    ],
    metadata={"category": "docs"},
    distance_threshold=0.72,
)

customer_support = Route(
    name="customer_support",
    references=[
        "how to respond to a complaint",
        "refund policy explanation",
        "triage steps for a ticket",
        "escalation guidelines",
        "roteiro de atendimento",
    ],
    metadata={"category": "support"},
    distance_threshold=0.73,
)

hr = Route(
    name="hr",
    references=[
        "vacation policy",
        "how to request leave",
        "benefits overview",
        "onboarding checklist",
        "policy compliance reminder",
        "folga e férias",
    ],
    metadata={"category": "hr"},
    distance_threshold=0.73,
)

legal = Route(
    name="legal",
    references=[
        "nda template",
        "contract review checklist",
        "privacy policy summary",
        "intellectual property basics",
        "dados pessoais e LGPD",
    ],
    metadata={"category": "legal"},
    distance_threshold=0.76,
)

shopping = Route(
    name="shopping",
    references=[
        "best laptops under 1000",
        "compare these products",
        "which phone should I buy",
        "dicas para economizar",
        "qual o melhor custo-benefício",
    ],
    metadata={"category": "shopping"},
    distance_threshold=0.72,
)

math_calc = Route(
    name="math",
    references=[
        "solve this math problem",
        "calculate percentage",
        "explain statistics concept",
        "derivative of a function",
        "probability basics",
    ],
    metadata={"category": "math"},
    distance_threshold=0.72,
)

general_chitchat = Route(
    name="general",
    references=[
        "how are you",
        "tell me a joke",
        "what can you do",
        "converse comigo",
        "small talk",
    ],
    metadata={"category": "general"},
    distance_threshold=0.70,
)

personal = Route(
    name="personal",
    references=[
        "what is my name",
        "where do I work",
        "my preferences",
        "minhas preferências",
    ],
    metadata={"category": "personal-stuff"},
    distance_threshold=0.71,
)

# --- Build router (uses OpenAI embeddings; no heavy deps) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_vectorizer = OpenAITextVectorizer(model=EMBED_MODEL)

# Lista única com todas as rotas:
_ALL_ROUTES = [
    technology, sports, entertainment, finance, health, travel, education, food,
    coding, devops, documentation, customer_support, hr, legal, shopping, math_calc,
    general_chitchat, personal
]

_router = SemanticRouter(
    name="topic-router",
    redis_url=REDIS_URL,
    vectorizer=_vectorizer,
    routes=_ALL_ROUTES,
    overwrite=False,
)

def _build_router(overwrite: bool = False) -> SemanticRouter:
    return SemanticRouter(
        name="topic-router",
        redis_url=REDIS_URL,
        vectorizer=_vectorizer,
        routes=_ALL_ROUTES,
        overwrite=overwrite,
    )

def init_router(overwrite: bool = False) -> None:
    global _router
    if overwrite:
        _router = _build_router(overwrite=True)
    _ = _router("health check")
    if overwrite:
        _router = _build_router(overwrite=False)

def route_query(text: str) -> Dict[str, Any]:
    m = _router(text)
    return {
        "name": m.name if getattr(m, "name", None) else None,
        "distance": float(getattr(m, "distance", 0.0)) if getattr(m, "name", None) else None,
        "metadata": getattr(m, "metadata", {}) or {},
    }