# Redis CESC + RAG Chatbot Demo

## Architecture

![Redis CESC Architecture](images/cesc_redis.png)

The diagram above shows how profile, short-term memory, long-term memory,
semantic router, semantic cache, and RAG interact in the workflow.

## Overview
This project is a **demo application** showcasing how to use **Redis** as the backbone for building
a Context-Enabled Semantic Cache (CESC) combined with Retrieval-Augmented Generation (RAG).

It integrates:
- **Redis** with RedisVL for vector search and semantic routing
- **Short-term memory (STM)** stored as Redis Lists with TTL
- **Long-term memory (LTM)** stored as RedisJSON arrays
- **User profile metadata** stored as Redis Hashes
- **Semantic Router** for topic classification using RedisVL routes
- **Semantic Cache** for caching generic answers with embeddings
- **RAG knowledge base** with chunked documents (e.g. PDFs)
- **LLMs** (OpenAI GPT models) for generating answers
- **Gradio UI** for interaction

The workflow ensures **fast, personalized, and context-aware answers** by combining cache hits with lightweight personalization
and cache misses with RAG + premium models.

---

## Features
- Upload PDFs → automatically chunked, embedded, and ingested into Redis for RAG.
- Semantic Router routes queries to relevant categories (tech, sports, finance, etc.).
- Semantic Cache improves performance and cost by reusing answers for semantically similar queries.
- STM and LTM provide conversational context and persistent memory.
- Persona/Mode selector in the UI (strict RAG, creative, analyst, support agent).
- Gradio-based web interface with chat history, file ingestion, and metadata controls.

---

## Architecture
1. **User Input (Gradio UI)** → user question + preferences.
2. **Profile Update (Redis Hash)** → tone, locale, persona, etc.
3. **STM (Redis List)** → recent conversation, expiring with TTL.
4. **LTM (RedisJSON)** → durable facts about the user.
5. **Semantic Router** → classifies query into a topic route.
6. **Semantic Cache** → checks if a semantically similar question already exists.
   - **Cache Hit:** cheap model personalizes the cached answer with profile/STM/RAG context.
   - **Cache Miss:** RAG retrieves relevant KB chunks, premium model generates generic answer, stored in cache.
7. **Final Answer** → returned to UI, appended to STM, possible promotion of facts into LTM.

---

## Requirements
- Python 3.12+
- Redis Stack (with RedisJSON, RediSearch)
- OpenAI API key
- Dependencies listed in `requirements.txt`

---

## Running Locally

### 1. Clone and install dependencies
```bash
git clone <this-repo>
cd redis-context-enabled-semantic-cache-llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
REDIS_URL=redis://localhost:6379/0
```

### 3. Run Redis
You can run Redis locally or with Docker Compose:
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:7.2.0-v14
```

### 4. Start the app
```bash
python -m src.app
```
Access at http://localhost:7860

---

## Docker

You can build and run the demo using the included `Dockerfile`:

```bash
docker build -t redis-cesc-rag-demo .
docker run -p 7860:7860 --env-file .env-docker redis-cesc-rag-demo
```

---

## License
MIT License

This demo is for **educational and demonstration purposes** only. Do not use in production without security, monitoring,
and performance adjustments.
