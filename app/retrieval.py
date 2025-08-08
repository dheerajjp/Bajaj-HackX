
from .embeddings import embed_query
from .config import TOP_K, SIMILARITY_THRESHOLD, MAX_CONTEXT_CHARS

def retrieve(store, question: str, id_to_text: dict):
    qvec = embed_query(question)
    raw = store.search(qvec, top_k=TOP_K)
    filtered = [(cid, score, meta) for cid, score, meta in raw if score >= SIMILARITY_THRESHOLD]
    contexts = []
    for cid, score, meta in filtered:
        text = (id_to_text.get(cid) or "")[:MAX_CONTEXT_CHARS]
        contexts.append({
            "id": cid,
            "score": float(score),
            "text": text,
            "metadata": meta
        })
    if not contexts and raw:
        for cid, score, meta in raw[:2]:
            text = (id_to_text.get(cid) or "")[:MAX_CONTEXT_CHARS]
            contexts.append({
                "id": cid, "score": float(score), "text": text, "metadata": meta
            })
    return contexts
