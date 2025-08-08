
import os, hashlib
from fastapi import FastAPI, Body, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from .config import API_PREFIX
from .security import require_bearer
from .models import RunRequest, RunResponse, Answer
from . import ingest
from .vectorstore import upsert_chunks
from .retrieval import retrieve
from .reasoner import call_llm

app = FastAPI(title="LLM Query Retrieval API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(API_PREFIX + "/health")
def health():
    return {"status": "ok"}

def _doc_index_name(urls):
    m = hashlib.sha256(("|".join(sorted(urls))).encode("utf-8")).hexdigest()[:10]
    return f"idx_{m}"

@app.post(API_PREFIX + "/hackrx/run", response_model=RunResponse)
def run_submission(req: RunRequest = Body(...), _: bool = Depends(require_bearer), debug: bool = Query(False)):
    if isinstance(req.documents, str):
        doc_urls = [req.documents]
    elif isinstance(req.documents, list):
        doc_urls = req.documents
    else:
        raise ValueError("`documents` must be a string URL or list of URLs")

    all_chunks = []
    id_to_text = {}
    for url in doc_urls:
        chunks = ingest.ingest(url)
        for ch in chunks:
            all_chunks.append({"id": ch.id, "text": ch.text, "metadata": ch.metadata})
            id_to_text[ch.id] = ch.text

    index_name = _doc_index_name(doc_urls)
    store = upsert_chunks(index_name, all_chunks)

    answers = []
    traces = []

    for q in req.questions:
        contexts = retrieve(store, q, id_to_text)
        llm_out = call_llm(q, contexts)
        answers.append(llm_out.get("answer", "Not explicitly stated."))

        if debug:
            cited = set(llm_out.get("citations", []))
            traces.append(Answer(
    answer=llm_out.get("answer", ""),
    reasoning=llm_out.get("reasoning", ""),
    confidence=llm_out.get("confidence", None),
    source_clauses=[{
        "id": c["id"],
        "score": c["score"],
        "metadata": c["metadata"]
    } for c in contexts if c["id"] in cited]
).model_dump())



    resp = {"answers": answers}
    if debug:
        resp["traces"] = traces
    return resp
