
import numpy as np
from typing import List
from .config import EMBEDDER, OPENAI_API_KEY

_sentence_model = None
_openai_ready = None

def _ensure_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def _ensure_openai():
    global _openai_ready
    if _openai_ready is None:
        _openai_ready = bool(OPENAI_API_KEY)
    return _openai_ready

def embed_texts(texts: List[str]) -> np.ndarray:
    if EMBEDDER == "openai" and _ensure_openai():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            vecs = [d.embedding for d in resp.data]
            return np.array(vecs, dtype="float32")
        except Exception:
            import openai
            openai.api_key = OPENAI_API_KEY
            resp = openai.Embedding.create(model="text-embedding-ada-002", input=texts)
            vecs = [d["embedding"] for d in resp["data"]]
            return np.array(vecs, dtype="float32")
    model = _ensure_sentence_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
