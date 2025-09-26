# src/ui/gradio_app.py
from __future__ import annotations

import uuid
from typing import List, Dict, Union, Any

import gradio as gr

from src.config import (
    DEFAULT_USER_ID,
    EMBED_MODEL,
    PREMIUM_LLM,
    CHEAP_LLM,
)
from src.memory.short_term import append_short_term
from src.profiles.user_profile import upsert_profile
from src.workflows import answer_one
from src.rag.pdf_ingest import ingest_uploaded_pdfs

# Optional: best-effort LTM writes; don't let UI fail the chat if Redis hiccups
try:
    from src.memory.long_term import upsert_fact
except Exception:  # pragma: no cover
    def upsert_fact(*args, **kwargs):  # type: ignore
        return None

SESSION_ID = str(uuid.uuid4())


def chat_fn(
    message: str,
    history: List[Dict[str, str]],
    tone: str,
    locale: str,
    role: str,
    interest: str,
    persona: str,
):
    """
    Chatbot handler: history is a list of dicts in 'messages' format:
    [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    Must return (updated_history, cleared_textbox).
    """
    message = (message or "").strip()
    if not message:
        return history or [], ""

    # 1) Persist user profile (Hash)
    upsert_profile(
        DEFAULT_USER_ID,
        {
            "tone": tone,
            "locale": locale,
            #"name": "Janine Cerioni",  # demo identity; wire from UI if needed
            "role": role,
            "interest": interest,
            "company": "Redis",
            "persona": persona,
            "mode": persona,  # alias
        },
    )

    # 2) Mirror key profile fields into LTM as structured facts (safe/no-op on failure)
    try:
        upsert_fact(DEFAULT_USER_ID, "persona", persona, source="ui", confidence=0.9)
        upsert_fact(DEFAULT_USER_ID, "locale", locale, source="ui", confidence=0.9)
        upsert_fact(DEFAULT_USER_ID, "tone", tone, source="ui", confidence=0.9)
        if role:
            upsert_fact(DEFAULT_USER_ID, "role", role, source="ui", confidence=0.8)
        if interest:
            upsert_fact(DEFAULT_USER_ID, "interest", interest, source="ui", confidence=0.7)
    except Exception:
        # Don't let LTM failures break chat
        pass

    # 3) Short-term memory: append user turn
    try:
        append_short_term(SESSION_ID, "user", message)
    except Exception:
        pass

    # 4) Orchestrate full flow and get reply
    reply = answer_one(DEFAULT_USER_ID, SESSION_ID, message)

    # 5) Short-term memory: append assistant turn
    try:
        append_short_term(SESSION_ID, "assistant", reply)
    except Exception:
        pass

    # 6) Update Chatbot history
    new_history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, ""


def kb_ingest_fn(files: List[Union[str, bytes, Dict[str, Any]]]) -> str:
    """
    Accepts payload from gr.Files in multiple shapes:
      - type="filepath"  -> List[str] (paths)
      - type="binary"    -> List[bytes] (or dicts with name/data/tempfile)
    """
    if not files:
        return "No PDFs selected."
    n_docs, n_chunks, ids = ingest_uploaded_pdfs(files)
    if n_docs == 0:
        return "No text extracted. Are these image-only PDFs?"
    return f"✅ Ingested {n_docs} PDF(s), ~{n_chunks} chunk(s).\nIDs: {', '.join(ids)}"


def clear_chat():
    # Only clears the UI; STM in Redis expires via TTL.
    return [], ""


def build() -> gr.Blocks:
    with gr.Blocks(title="Redis CESC + RAG Demo") as demo:
        # ===== HEADER =====
        gr.Markdown(
            "## Redis CESC + RAG — Demo\n"
            "Assistente genérico com **Vector KB**, **memória** (curta e longa), "
            "**semantic cache**, **routing** e **upload de PDFs**.\n"
        )

        # ===== MODEL CARD =====
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    f"**Embedding model:** `{EMBED_MODEL}`  \n"
                    f"**Premium LLM (miss path):** `{PREMIUM_LLM}`  \n"
                    f"**Cheap LLM (cache hit):** `{CHEAP_LLM}`"
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    f"**User ID:** `{DEFAULT_USER_ID}`  \n"
                    f"**Session:** `{SESSION_ID}`  \n"
                    "*(IDs só para referência na demo)*"
                )

        # ===== USER CONTROLS =====
        gr.Markdown("### Preferências do Usuário")
        with gr.Row():
            tone = gr.Dropdown(
                ["concise", "friendly", "formal"], value="concise", label="Tone"
            )
            locale = gr.Dropdown(["pt-BR", "en-US"], value="pt-BR", label="Locale")
            role = gr.Dropdown(
                ["analyst", "customer", "manager", "financial advisor"],
                value="financial advisor",
                label="Role",
            )
            persona = gr.Dropdown(
                [
                    "rag_strict",
                    "creative_helper",
                    "analyst",
                    "support_agent",
                ],
                value="rag_strict",
                label="Persona/Mode",
            )

        with gr.Row():
            interest = gr.Textbox(
                placeholder="User interests (comma separated)",
                value="investments, markets",
                label="Interests",
            )

        # ===== PDF UPLOAD =====
        with gr.Accordion("Add PDFs to Knowledge Base", open=False):
            # Use type="filepath" for stable server-side reading
            pdfs = gr.Files(label="Upload PDFs", file_types=[".pdf"], type="filepath")
            with gr.Row():
                ingest_btn = gr.Button("Ingest into KB")
                ingest_status = gr.Markdown("")
            ingest_btn.click(fn=kb_ingest_fn, inputs=[pdfs], outputs=[ingest_status])

        # ===== CHAT AREA =====
        gr.Markdown("### Chat")
        chat = gr.Chatbot(height=460, type="messages", label="Assistant")
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Pergunte sobre documentação, código, viagens, esportes… (ou faça upload de PDFs acima)",
                scale=4,
            )
            send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear", variant="secondary", scale=1)

        # Wire inputs (Chatbot uses 'messages' format)
        send.click(
            fn=chat_fn,
            inputs=[msg, chat, tone, locale, role, interest, persona],
            outputs=[chat, msg],
        )
        msg.submit(
            fn=chat_fn,
            inputs=[msg, chat, tone, locale, role, interest, persona],
            outputs=[chat, msg],
        )
        clear.click(fn=clear_chat, outputs=[chat, msg])

    return demo