# src/ui/gradio_app.py
import uuid
import gradio as gr
from typing import List, Dict

from src.config import (
    DEFAULT_USER_ID,
    EMBED_MODEL,
    PREMIUM_LLM,
    CHEAP_LLM,
)
from src.memory.short_term import append_short_term
from src.memory.long_term import add_longterm_fact
from src.profiles.user_profile import upsert_profile
from src.workflows import answer_one
from src.rag.pdf_ingest import ingest_uploaded_pdfs

SESSION_ID = str(uuid.uuid4())


def chat_fn(
    message: str,
    history: List[Dict[str, str]],
    tone: str,
    locale: str,
    role: str,
    interest: str,
    persona: str,   # <-- novo
):
    """
    history is a list of dicts in messages format:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    Must return (updated_history, cleared_textbox)
    """
    # Store/refresh user profile metadata (Hash)
    upsert_profile(
        DEFAULT_USER_ID,
        {
            "tone": tone,
            "locale": locale,
            "name": "Gabriel Cerioni",             # demo identity (pode vir da UI se quiser)
            "role": role,
            "interest": interest,
            "company": "Bradesco",
            "persona": persona,                    # <-- gravamos a persona escolhida
            "mode": persona,                       # alias comum
        },
    )

    # Persist some facts in long-term memory (RedisJSON array) – demo-only
    add_longterm_fact(DEFAULT_USER_ID, f"persona={persona}")
    add_longterm_fact(DEFAULT_USER_ID, f"locale={locale}")
    add_longterm_fact(DEFAULT_USER_ID, f"tone={tone}")

    # Short-term memory: append user turn
    append_short_term(SESSION_ID, "user", message)

    # Orchestrate CESC flow and get reply
    reply = answer_one(DEFAULT_USER_ID, SESSION_ID, message)

    # Short-term memory: append assistant turn
    append_short_term(SESSION_ID, "assistant", reply)

    # Update messages-format history for Gradio Chatbot
    new_history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, ""  # clear the textbox


def kb_ingest_fn(files: List[Dict]) -> str:
    """
    Accepts payload from gr.Files(type='binary' or 'filepath').
    """
    if not files:
        return "No PDFs selected."
    n_docs, n_chunks, ids = ingest_uploaded_pdfs(files)
    if n_docs == 0:
        return "No text extracted. Are these image-only PDFs?"
    return f"✅ Ingested {n_docs} PDF(s), ~{n_chunks} chunk(s).\nIDs: {', '.join(ids)}"


def clear_chat():
    # Apenas limpa a interface; STM expira por TTL no Redis.
    return [], ""


def build():
    with gr.Blocks(title="Redis CESC + RAG Demo") as demo:
        # ===== HEADER =====
        gr.Markdown(
            "## Redis CESC + RAG — Demo\n"
            "Assistente genérico com **Vector KB**, **memória** (curta e longa), **semantic cache**, **routing** e **upload de PDFs**.\n"
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
            persona = gr.Dropdown(                   # <-- nova persona/mode
                [
                    "rag_strict",                   # usa apenas contexto, sem alucinar
                    "creative_helper",              # escreve com mais liberdade
                    "analyst",                      # foco em steps/bullets/justificativas
                    "support_agent",                # tom empático e procedural
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