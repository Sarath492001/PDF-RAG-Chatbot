"""Stage 1 - PDF -> clean, page-structured text.

* Native text via PyMuPDF (fast, keeps block bounding boxes).
* OCR fallback via Tesseract for scanned pages / pages dominated by an embedded image.
* Header/footer/page-number removal (repeated-line detection across the document).
* Unicode + hyphenation normalisation, per-page language detection.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
from langdetect import DetectorFactory, detect

from . import config as C

DetectorFactory.seed = 0  # deterministic language detection
EDGE_FRAC = 0.10          # top/bottom 10 % of a page = header/footer zone
PAGE_NUM_RE = re.compile(r"^\W*(page\s*)?#+(\s*(of|/)\s*#+)?\W*$")


def pdf_id_for(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\s*\n\s*(?=[a-z])", "", s)                    # re-join hyphenated line breaks
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u00ad]", "", s)    # control chars / soft hyphen
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def norm_key(s: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", s.strip().lower()))


def _ocr_page(page) -> str:
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(dpi=C.OCR_DPI, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=C.OCR_LANG)


def _init_worker():
    os.environ["OMP_THREAD_LIMIT"] = "1"  # one Tesseract thread per worker process


def _extract_range(path: str, start: int, end: int) -> list[dict]:
    doc = fitz.open(path)
    out = []
    for pno in range(start, end):
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        blocks, native_chars = [], 0
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0 or not b[4].strip():      # skip image blocks / empty
                continue
            zone = "header" if b[3] <= H * EDGE_FRAC else "footer" if b[1] >= H * (1 - EDGE_FRAC) else "body"
            blocks.append({"bbox": [round(v, 1) for v in b[:4]], "text": b[4], "zone": zone})
            native_chars += len(b[4].strip())

        cov = 0.0
        try:
            for im in page.get_image_info():
                x0, y0, x1, y1 = im["bbox"]
                cov += max(0, x1 - x0) * max(0, y1 - y0)
            cov /= max(W * H, 1)
        except Exception:
            pass

        method, error = "native", None
        scanned = native_chars < C.MIN_NATIVE_CHARS or (cov > 0.6 and native_chars < 300)
        if C.OCR_ENABLED and scanned:
            try:
                text = _ocr_page(page)
                if len(text.strip()) > native_chars:
                    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
                    blocks = []
                    for i, p in enumerate(paras):
                        zone = "header" if (i == 0 and len(p) < 100) else "footer" if (i == len(paras) - 1 and len(p) < 100) else "body"
                        blocks.append({"bbox": [0.0, 0.0, round(W, 1), round(H, 1)], "text": p, "zone": zone})
                    method = "ocr"
            except Exception as e:  # tesseract missing etc. -> keep native text
                error = f"{type(e).__name__}: {e}"
        out.append({"page": pno + 1, "method": method, "w": W, "h": H, "blocks": blocks, "error": error})
    doc.close()
    return out


def extract_pdf(path, workers: int | None = None, progress=None) -> dict:
    """Return {'filename','n_pages','ocr_pages','pages':[{'page','method','lang','blocks':[{'bbox','text'}]}]}"""
    path = str(path)
    workers = C.WORKERS if workers is None else workers
    doc = fitz.open(path)
    if doc.needs_pass:
        raise ValueError(f"{os.path.basename(path)} is password-protected")
    n = doc.page_count
    doc.close()

    ranges = [(s, min(s + C.PAGES_PER_TASK, n)) for s in range(0, n, C.PAGES_PER_TASK)]
    results: dict[int, list] = {}
    done = 0
    if workers <= 1:
        for s, e in ranges:
            results[s] = _extract_range(path, s, e)
            done += e - s
            progress and progress(done, n)
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
            futs = {ex.submit(_extract_range, path, s, e): (s, e) for s, e in ranges}
            for f in as_completed(futs):
                s, e = futs[f]
                results[s] = f.result()
                done += e - s
                progress and progress(done, n)
    pages = [p for s in sorted(results) for p in results[s]]

    # ---- header / footer / page-number removal (document-level repeated lines) ----
    cnt = Counter()
    for pg in pages:
        cnt.update({norm_key(b["text"]) for b in pg["blocks"] if b["zone"] != "body"})
    thr = max(3, int(0.25 * n))
    boiler = {k for k, v in cnt.items() if v >= thr}

    for pg in pages:
        kept = []
        for b in pg["blocks"]:
            key = norm_key(b["text"])
            if b["zone"] != "body" and (key in boiler or PAGE_NUM_RE.match(key)):
                continue
            text = clean_text(b["text"])
            if len(text) >= 2:
                kept.append({"bbox": b["bbox"], "text": text})
        pg["blocks"] = kept
        sample = " ".join(b["text"] for b in kept)[:1000]
        try:
            pg["lang"] = detect(sample) if len(sample) >= 40 else "und"
        except Exception:
            pg["lang"] = "und"

    return {
        "filename": os.path.basename(path),
        "n_pages": n,
        "ocr_pages": sum(p["method"] == "ocr" for p in pages),
        "ocr_errors": sorted({p["error"] for p in pages if p.get("error")}),
        "pages": pages,
    }
