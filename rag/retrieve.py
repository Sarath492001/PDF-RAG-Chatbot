"""Stage 4/5 - dense ANN retrieval (HNSW) + optional cross-encoder rerank."""
from __future__ import annotations

import time

from . import config as C
from .store import get_embedder, get_store


class Retriever:
    def __init__(self):
        self.emb = get_embedder()
        self.store = get_store()
        self.reranker = None
        if C.USE_RERANK:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(C.RERANK_MODEL, max_length=512, device=self.emb.device)

    def retrieve(self, query: str, top_k=None, rerank=None, files=None) -> dict:
        """Returns {'hits': [ranked hits...], 'timings': {...ms}}. hits[i] has dense_score and rerank_score."""
        top_k = top_k or C.TOP_K
        use_rr = (C.USE_RERANK if rerank is None else rerank) and self.reranker is not None

        t0 = time.perf_counter()
        qv = self.emb.embed_query(query)
        t1 = time.perf_counter()
        hits = self.store.search(qv, top_k, files)
        t2 = time.perf_counter()
        for h in hits:
            h["rerank_score"] = None
        if use_rr and hits:
            scores = self.reranker.predict([(query, h["text"]) for h in hits], batch_size=32, show_progress_bar=False)
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        t3 = time.perf_counter()
        for i, h in enumerate(hits, 1):
            h["rank"] = i
        return {"hits": hits, "timings": {
            "embed_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000,
            "rerank_ms": (t3 - t2) * 1000, "retrieval_ms": (t3 - t0) * 1000}}
