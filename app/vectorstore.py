
import os, json
import numpy as np
from typing import List, Dict, Any, Tuple
from .config import STORAGE_DIR, VECTOR_MODE, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX
from .embeddings import embed_texts

class BaseVectorStore:
    def add(self, ids: List[str], vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        raise NotImplementedError
    def search(self, vector: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

class FAISSStore(BaseVectorStore):
    def __init__(self, name: str):
        import faiss
        self.name = name
        self.dir = os.path.join(STORAGE_DIR, "faiss")
        os.makedirs(self.dir, exist_ok=True)
        self.index_path = os.path.join(self.dir, f"{name}.index")
        self.meta_path = os.path.join(self.dir, f"{name}.meta.json")

        self.index = None
        self.meta: Dict[str, Dict[str, Any]] = {}
        self._load_if_exists()

    def _load_if_exists(self):
        import faiss
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def _save(self):
        import faiss
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

    def add(self, ids: List[str], vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        import faiss
        vectors = vectors.astype("float32")
        dim = vectors.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        self.index.add(vectors)
        for i, mid in enumerate(ids):
            self.meta[mid] = metadatas[i]
        self._save()

    def search(self, vector: np.ndarray, top_k: int):
        import faiss
        if self.index is None:
            return []
        vector = vector.astype("float32").reshape(1, -1)
        scores, idx = self.index.search(vector, top_k)
        results = []
        keys = list(self.meta.keys())
        for row, score in zip(idx[0], scores[0]):
            if row < 0 or row >= len(keys):
                continue
            mid = keys[row]
            results.append((mid, float(score), self.meta[mid]))
        return results

class PineconeStore(BaseVectorStore):
    def __init__(self, name: str):
        import pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index_name = PINECONE_INDEX or (name + "-index")
        existing = [i.name for i in pinecone.list_indexes()]
        if self.index_name not in existing:
            pinecone.create_index(self.index_name, dimension=384, metric="cosine")
        self.index = pinecone.Index(self.index_name)

    def add(self, ids: List[str], vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        vecs = vectors.tolist()
        to_upsert = list(zip(ids, vecs, metadatas))
        self.index.upsert(vectors=to_upsert)

    def search(self, vector: np.ndarray, top_k: int):
        res = self.index.query(vector=vector.tolist(), top_k=top_k, include_metadata=True)
        out = []
        for m in res.matches or []:
            out.append((m.id, float(m.score), m.metadata or {}))
        return out

def get_store(name: str) -> BaseVectorStore:
    if VECTOR_MODE == "pinecone" and PINECONE_API_KEY:
        return PineconeStore(name=name)
    return FAISSStore(name=name)

def upsert_chunks(name: str, chunks: List[Dict[str, Any]]):
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    vecs = embed_texts(texts)
    store = get_store(name)
    store.add(ids, vecs, metas)
    return store
