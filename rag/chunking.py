"""Stage 2 - deterministic token-based chunking with overlap + per-chunk metadata."""
from __future__ import annotations

import hashlib
import re
from collections import namedtuple
from functools import lru_cache

from . import config as C

Unit = namedtuple("Unit", "text n page bbox lang method")
_SENT = re.compile(r"(?<=[.!?])\s+")


class Chunker:
    def __init__(self, tokenizer_name=None, chunk_tokens=None, overlap_ratio=None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name or C.EMBED_MODEL, use_fast=True)
        self.tok.model_max_length = 10**7  # silence "sequence too long" warnings; we enforce limits ourselves
        self.budget = chunk_tokens or C.CHUNK_TOKENS
        self.overlap = int(self.budget * (C.OVERLAP_RATIO if overlap_ratio is None else overlap_ratio))

    def ntok(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    # --- split oversize text into <= budget pieces (sentences first, hard token windows last) ---
    def _pieces(self, text: str):
        if self.ntok(text) <= self.budget:
            yield text
            return
        for sent in _SENT.split(text):
            if not sent.strip():
                continue
            if self.ntok(sent) <= self.budget:
                yield sent
            else:
                ids = self.tok.encode(sent, add_special_tokens=False)
                for i in range(0, len(ids), self.budget):
                    yield self.tok.decode(ids[i:i + self.budget])

    def _units(self, doc: dict):
        for pg in doc["pages"]:
            for b in pg["blocks"]:
                for piece in self._pieces(b["text"]):
                    yield Unit(piece, self.ntok(piece), pg["page"], b["bbox"], pg.get("lang", "und"), pg["method"])

    def chunk_document(self, doc: dict) -> list[dict]:
        chunks, cur, fresh = [], [], 0

        def flush():
            idx = len(chunks)
            first = cur[0]
            same = [u.bbox for u in cur if u.page == first.page]
            bbox = [min(b[0] for b in same), min(b[1] for b in same), max(b[2] for b in same), max(b[3] for b in same)]
            chunks.append({
                "id": hashlib.sha1(f"{doc['pdf_id']}:{idx}".encode()).hexdigest()[:20],
                "text": "\n".join(u.text for u in cur),
                "meta": {
                    "pdf_id": doc["pdf_id"], "filename": doc["filename"],
                    "page": first.page, "page_end": cur[-1].page,
                    "bbox": ",".join(f"{v:.1f}" for v in bbox),
                    "chunk_idx": idx, "n_tokens": sum(u.n for u in cur),
                    "lang": first.lang, "method": first.method,
                },
            })

        cur_tok = 0
        for u in self._units(doc):
            if cur and cur_tok + u.n > self.budget:
                flush()
                fresh = 0
                keep, t = [], 0                      # carry trailing units as overlap
                for x in reversed(cur):
                    if t + x.n > self.overlap:
                        break
                    keep.insert(0, x)
                    t += x.n
                if t + u.n > self.budget:
                    keep, t = [], 0
                cur, cur_tok = keep, t
            cur.append(u)
            cur_tok += u.n
            fresh += 1
        if fresh:
            flush()
        return chunks


@lru_cache(maxsize=1)
def get_chunker() -> Chunker:
    return Chunker()
