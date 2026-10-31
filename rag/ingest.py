"""Ingestion orchestrator:  PDF -> extract(+OCR) -> clean -> chunk -> embed -> HNSW index.

CLI:
    python -m rag.ingest --dir data/pdfs            # ingest every new PDF
    python -m rag.ingest --file path/to/x.pdf
    python -m rag.ingest --dir data/pdfs --force    # re-chunk + re-embed (re-uses cached OCR/extraction)
    python -m rag.ingest --dir data/pdfs --force --no-cache   # also redo extraction/OCR
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .chunking import get_chunker
from .extract import extract_pdf, pdf_id_for
from .store import get_embedder, get_store, load_manifest, save_manifest

# share of the overall progress bar given to each stage
W = {"extract": (0.0, 0.40), "chunk": (0.40, 0.45), "embed": (0.45, 0.95), "index": (0.95, 1.0)}


def ingest_pdf(path, progress=None, force=False, workers=None, use_cache=True) -> dict:
    """progress(stage:str, overall_fraction:float, message:str)"""
    path = Path(path)
    pid = pdf_id_for(path)
    man = load_manifest()
    if pid in man and not force:
        return {"filename": path.name, "skipped": True, **man[pid]}

    def p(stage, frac, msg=""):
        lo, hi = W[stage]
        progress and progress(stage, lo + (hi - lo) * frac, msg)

    secs = {}
    # 1) extract (+OCR, cleaning, language detection) - cached so OCR is never repeated
    t = time.perf_counter()
    cache = C.PROCESSED_DIR / "pages" / f"{pid}.json.gz"
    if use_cache and cache.exists():
        doc = json.loads(gzip.open(cache, "rt", encoding="utf-8").read())
        p("extract", 1.0, "loaded cached extraction")
    else:
        doc = extract_pdf(path, workers, progress=lambda d, n: p("extract", d / n, f"page {d}/{n}"))
        with gzip.open(cache, "wt", encoding="utf-8") as f:
            json.dump(doc, f)
    doc["pdf_id"] = pid
    secs["extract"] = time.perf_counter() - t

    # 2) chunk
    t = time.perf_counter()
    p("chunk", 0.0, "chunking")
    chunks = get_chunker().chunk_document(doc)
    with open(C.PROCESSED_DIR / "chunks" / f"{pid}.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    secs["chunk"] = time.perf_counter() - t
    p("chunk", 1.0, f"{len(chunks)} chunks")

    # 3) embed
    t = time.perf_counter()
    vecs = get_embedder().embed_docs([c["text"] for c in chunks],
                                     progress=lambda d, n: p("embed", d / n, f"embedded {d}/{n} chunks"))
    secs["embed"] = time.perf_counter() - t

    # 4) index (HNSW)
    t = time.perf_counter()
    store = get_store()
    if force:
        store.delete_pdf(pid)
    p("index", 0.0, "writing to vector DB")
    store.add(chunks, vecs)
    secs["index"] = time.perf_counter() - t
    p("index", 1.0, "done")

    langs = sorted({pg["lang"] for pg in doc["pages"] if pg["lang"] != "und"})
    entry = {"filename": path.name, "pages": doc["n_pages"], "ocr_pages": doc["ocr_pages"], "chunks": len(chunks),
             "languages": langs, "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    man[pid] = entry
    save_manifest(man)
    return {**entry, "skipped": False, "seconds": {k: round(v, 1) for k, v in secs.items()},
            "ocr_errors": doc.get("ocr_errors", [])}


def ingest_dir(directory=None, **kw) -> list[dict]:
    pdfs = sorted(Path(directory or C.PDF_DIR).glob("*.pdf"))
    return [ingest_pdf(p, **kw) for p in pdfs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(C.PDF_DIR))
    ap.add_argument("--file")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    files = [Path(a.file)] if a.file else sorted(Path(a.dir).glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found in {a.dir}")
    last = {"s": ""}

    def cb(stage, frac, msg):
        line = f"\r  [{stage:7s}] {frac*100:5.1f}%  {msg:<40}"
        if line != last["s"]:
            print(line, end="", flush=True)
            last["s"] = line

    t0 = time.perf_counter()
    tot_pages = tot_chunks = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}")
        r = ingest_pdf(f, progress=cb, force=a.force, workers=a.workers, use_cache=not a.no_cache)
        print()
        if r["skipped"]:
            print("  already ingested - skipped (use --force to redo)")
        else:
            print(f"  pages={r['pages']} ocr_pages={r['ocr_pages']} chunks={r['chunks']} langs={r['languages']} time={r['seconds']}")
            for e in r["ocr_errors"]:
                print(f"  WARNING OCR failed on some pages: {e}  (is Tesseract installed?)")
        tot_pages += r["pages"]
        tot_chunks += r["chunks"]
    print(f"\nDone: {len(files)} PDFs, {tot_pages} pages, {tot_chunks} chunks in {time.perf_counter()-t0:.0f}s. "
          f"Vector DB now holds {get_store().count()} chunks.")


if __name__ == "__main__":
    main()
