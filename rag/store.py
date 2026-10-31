"""Stage 3/4 - embeddings + persistent HNSW vector index (ChromaDB) + ingestion manifest."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from . import config as C


class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = C.DEVICE or ("cuda" if torch.cuda.is_available()
                                   else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                   else "cpu")
        self.model = SentenceTransformer(C.EMBED_MODEL, device=self.device)
        self.model.max_seq_length = C.MAX_SEQ_LEN
        self.dim = self.model.get_sentence_embedding_dimension()
        self._qcache: dict[str, np.ndarray] = {}

    def embed_docs(self, texts, progress=None, batch_size=64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 256):
            part = [C.DOC_PREFIX + t for t in texts[i:i + 256]]
            out.append(self.model.encode(part, batch_size=batch_size, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False))
            progress and progress(min(i + 256, len(texts)), len(texts))
        return np.vstack(out).astype("float32") if out else np.zeros((0, self.dim), dtype="float32")

    def embed_query(self, q: str) -> np.ndarray:
        if q not in self._qcache:
            if len(self._qcache) > 512:
                self._qcache.clear()
            self._qcache[q] = self.model.encode([C.QUERY_PREFIX + q], normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=False)[0].astype("float32")
        return self._qcache[q]


class VectorStore:
    def __init__(self):
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(path=str(C.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(
            C.COLLECTION,
            metadata={"hnsw:space": "cosine", "hnsw:M": C.HNSW_M,
                      "hnsw:construction_ef": C.HNSW_EF_CONSTRUCTION, "hnsw:search_ef": C.HNSW_EF_SEARCH},
        )

    def add(self, chunks: list[dict], embeddings: np.ndarray, batch=2000):
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(ids=[c["id"] for c in part],
                            embeddings=embeddings[i:i + batch].tolist(),
                            documents=[c["text"] for c in part],
                            metadatas=[c["meta"] for c in part])

    def delete_pdf(self, pdf_id: str):
        self.col.delete(where={"pdf_id": pdf_id})

    def count(self) -> int:
        return self.col.count()

    def search(self, vec: np.ndarray, k: int, files: list[str] | None = None) -> list[dict]:
        where = {"filename": {"$in": files}} if files else None
        k = max(1, min(k, max(self.count(), 1)))
        r = self.col.query(query_embeddings=[vec.tolist()], n_results=k, where=where,
                           include=["documents", "metadatas", "distances"])
        return [{"id": i, "text": d, "meta": m, "dense_score": 1.0 - dist}
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore()


def load_manifest() -> dict:
    return json.loads(C.MANIFEST.read_text()) if C.MANIFEST.exists() else {}


def save_manifest(m: dict):
    C.MANIFEST.write_text(json.dumps(m, indent=2))
