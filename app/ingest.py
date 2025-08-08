
import os, re, hashlib, io, mimetypes, requests
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

DOC_TIMEOUT = 45

@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any]

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]

def _safe_text(txt: str) -> str:
    return re.sub(r'\s+', ' ', txt).strip()

def _guess_ext_from_ct(content_type: str) -> str:
    if not content_type:
        return ""
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
    return ext or ""

def fetch_document(url: str) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=DOC_TIMEOUT)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "")
    ext = _guess_ext_from_ct(content_type)
    if not ext:
        url_path = url.split("?")[0]
        ext = os.path.splitext(url_path)[1].lower()
    return r.content, ext

def parse_pdf(data: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    import fitz  # PyMuPDF
    out = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text:
                out.append((_safe_text(text), {"page": i}))
    return out

def parse_docx(data: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    import docx
    buf = io.BytesIO(data)
    document = docx.Document(buf)
    paras = []
    for p in document.paragraphs:
        t = p.text.strip()
        if t:
            paras.append((_safe_text(t), {}))
    chunks, curr, meta = [], [], {}
    chcount = 0
    for t, m in paras:
        curr.append(t)
        chcount += len(t)
        if chcount > 1500:
            chunks.append((" ".join(curr), meta))
            curr, chcount = [], 0
    if curr:
        chunks.append((" ".join(curr), meta))
    return chunks

def parse_email(data: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    import email
    msg = email.message_from_bytes(data)
    texts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ["text/plain", "text/html"]:
                payload = part.get_payload(decode=True) or b""
                try:
                    txt = payload.decode(part.get_content_charset() or "utf-8", errors="ignore")
                except Exception:
                    txt = payload.decode("utf-8", errors="ignore")
                txt = re.sub(r'<[^>]+>', ' ', txt)
                texts.append(_safe_text(txt))
    else:
        payload = msg.get_payload(decode=True) or b""
        txt = payload.decode("utf-8", errors="ignore")
        txt = re.sub(r'<[^>]+>', ' ', txt)
        texts.append(_safe_text(txt))
    joined = " ".join([t for t in texts if t])
    return [(joined, {})] if joined else []

def split_into_chunks(pairs: List[Tuple[str, Dict[str, Any]]], chunk_size=1200, overlap=150) -> List[Tuple[str, Dict[str, Any]]]:
    chunks = []
    for text, meta in pairs:
        text = text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            snippet = text[start:end]
            chunks.append((snippet, dict(meta)))
            if end == len(text):
                break
            start = end - overlap
    return chunks

def ingest(url: str) -> List[DocumentChunk]:
    data, ext = fetch_document(url)
    ext = (ext or "").lower()

    doc_id = _hash(url)

    if ext in [".pdf"]:
        pairs = parse_pdf(data)
    elif ext in [".docx"]:
        pairs = parse_docx(data)
    elif ext in [".eml", ".msg"]:
        pairs = parse_email(data)
    else:
        txt = data.decode("utf-8", errors="ignore")
        pairs = [(txt, {})]

    pairs = split_into_chunks(pairs, chunk_size=1200, overlap=150)

    chunks = []
    for i, (text, meta) in enumerate(pairs):
        chunk_id = f"{doc_id}_{i:04d}"
        meta.update({"doc_id": doc_id, "chunk_index": i, "source_url": url})
        chunks.append(DocumentChunk(id=chunk_id, text=text, metadata=meta))
    return chunks
