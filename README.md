
# LLM-Powered Intelligent Query–Retrieval System (FastAPI)

Production-ready baseline for the HackRX-style submission pipeline.

## 1) Features

- PDF/DOCX/Email ingestion (via URL)
- Smart chunking + metadata (page, doc id, source URL)
- Embeddings: Sentence-Transformers (local) or OpenAI
- Vector store: FAISS (default) or Pinecone
- Retrieval with similarity filtering + token-efficient context trimming
- LLM reasoning with JSON output scaffold
- Exact endpoint contract: POST /api/v1/hackrx/run -> { "answers": [...] }
- Bearer auth (token in instructions preloaded)

## 2) Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Health: GET http://localhost:8000/api/v1/health
- Endpoint: POST http://localhost:8000/api/v1/hackrx/run

## 3) Auth

```
Authorization: Bearer dc409f1e339103877a936ed2ef35093ce8d2623e7eb5a1bc58a7ee165ea44135
```

Override via env: `export ALLOWED_BEARER_TOKEN="your-token"`

## 4) Config

```bash
export VECTOR_MODE=faiss        # faiss | pinecone
export EMBEDDER=sentence        # sentence | openai
export OPENAI_API_KEY=          # set to enable OpenAI
export PINECONE_API_KEY=        # set to enable Pinecone
export PINECONE_ENV=us-west1-gcp
export PINECONE_INDEX=policy-index
export TOP_K=5
export SIMILARITY_THRESHOLD=0.65
export MAX_CONTEXT_CHARS=1200
```

## 5) Request/Response

Request:
```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
  "questions": [
    "What is the grace period...?",
    "What is the waiting period...?"
  ]
}
```

Response:
```json
{ "answers": ["...", "..."] }
```

Debug:
`POST /api/v1/hackrx/run?debug=true` includes `traces` with reasoning + citations.

## 6) Notes

- Without OPENAI_API_KEY, answers are heuristic (dev mode).
- Context is trimmed for token-efficiency.
- Modular code for reusability and extension.
