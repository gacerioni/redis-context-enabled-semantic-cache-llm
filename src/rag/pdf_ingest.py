from typing import List, Tuple, Dict, Any, Union
from pypdf import PdfReader
import hashlib, os
import io

from src.rag.ingest import Doc, upsert_kb
from src.utils.chunker import simple_chunk

def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _read_pdf_bytes(obj: Union[str, bytes, Dict[str, Any]]) -> tuple[str, bytes]:
    """
    Normalize different Gradio Files payload shapes into (name, bytes).
    Supports:
      - bytes (type='binary' in some Gradio versions)
      - str (file path when type='filepath')
      - dict with {'name','data'} (other Gradio versions)
      - dict with {'path'} or {'tempfile'}
    """
    # 1) direct bytes
    if isinstance(obj, (bytes, bytearray)):
        return ("upload.pdf", bytes(obj))

    # 2) filepath
    if isinstance(obj, str):
        path = obj
        name = os.path.basename(path) or "upload.pdf"
        with open(path, "rb") as fh:
            data = fh.read()
        return (name, data)

    # 3) dict payload
    if isinstance(obj, dict):
        name = obj.get("name") or "upload.pdf"
        data = obj.get("data")
        if data:
            return (name, data)
        # maybe only a path/tempfile was provided
        path = obj.get("path") or obj.get("tempfile")
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            return (os.path.basename(path) or name, data)

    # fallback
    return ("upload.pdf", b"")

def extract_text_from_pdf_bytes(content: bytes) -> str:
    """
    Robust text extraction from PDF bytes. Skips empty pages.
    """
    reader = PdfReader(io.BytesIO(content))
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")  # attempt empty-password decrypt
        except Exception:
            pass
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            pages.append(t)
    return "\n\n".join(pages).strip()

def build_doc_from_pdf(name: str, content: bytes) -> Doc:
    """
    Build a Doc object (doc_id, source, text) from a PDF.
    - doc_id: sha1 of bytes + sanitized filename (stable for de-dupe)
    - source: "upload/{filename}"
    """
    doc_hash = _sha1(content)[:12]
    base = os.path.splitext(os.path.basename(name) or "upload")[0]
    doc_id = f"pdf_{base}_{doc_hash}"
    text = extract_text_from_pdf_bytes(content)
    return Doc(
        doc_id=doc_id,
        source=f"upload/{os.path.basename(name) or 'upload.pdf'}",
        text=text
    )

def ingest_uploaded_pdfs(files: List[Union[str, bytes, Dict[str, Any]]]) -> Tuple[int, int, List[str]]:
    """
    Accepts Gradio Files payload in multiple shapes (bytes, filepaths, or dicts).
    Returns: (num_docs, total_chunks, doc_ids)
    """
    docs: List[Doc] = []
    total_chunks = 0
    doc_ids: List[str] = []

    for f in files or []:
        name, data = _read_pdf_bytes(f)
        if not data:
            continue

        doc = build_doc_from_pdf(name=name, content=data)
        if not doc.text:
            # likely an image-only PDF; skip silently (or log)
            continue

        # pre-count chunks (mirrors simple_chunk inside upsert)
        total_chunks += len(simple_chunk(doc.text))
        doc_ids.append(doc.doc_id)
        docs.append(doc)

    if docs:
        upsert_kb(docs)  # embeds + JSON.SET at root + pipeline

    return len(docs), total_chunks, doc_ids