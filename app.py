"""Streamlit demo UI:  streamlit run app.py
Tabs: Chat (streaming answer + citations + retrieval visualisation + latency) | Ingestion pipeline | Corpus"""
import math
import time

import fitz
import pymupdf
import pandas as pd
import streamlit as st

from rag import config as C
from rag.store import load_manifest

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model, reranker and vector index...")
def get_pipeline():
    from rag.generate import RAGPipeline
    p = RAGPipeline()
    p.warmup()
    return p


# @st.cache_data(show_spinner=False, max_entries=64)
# def page_png(filename: str, page: int) -> bytes | None:
#     try:
#         with fitz.open(C.PDF_DIR / filename) as d:
#             return d[page - 1].get_pixmap(dpi=75).tobytes("png")
#     except Exception:
#         return None
@st.cache_data(show_spinner=False, max_entries=64)
def page_png(filename: str, page: int) -> bytes | None:
    try:
        with pymupdf.open(C.PDF_DIR / filename) as d:
            return d[page - 1].get_pixmap(dpi=75).tobytes("png")
    except Exception:
        return None

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# def render_retrieval(hits, key):
#     st.markdown("**🔎 Retrieval visualisation** - ANN candidates (HNSW) re-ranked by the cross-encoder; ✅ = sent to the LLM")
#     df = pd.DataFrame([{
#         "#": h["rank"], "used": "✅" if h["in_context"] else "", "file": h["filename"],
#         "page": h["page"] if h["page"] == h["page_end"] else f"{h['page']}-{h['page_end']}",
#         "dense cos": round(h["dense_score"], 3),
#         "rerank p": None if h["rerank_score"] is None else round(sigmoid(h["rerank_score"]), 3),
#         "ocr": h["method"] == "ocr", "snippet": h["text"][:160].replace("\n", " ") + "...",
#     } for h in hits])
#     st.dataframe(df, hide_index=True, width="stretch", key=f"df_{key}")
#     chart = pd.DataFrame({"dense cosine": [h["dense_score"] for h in hits]},
#                          index=[f"#{h['rank']} {h['filename'][:18]} p{h['page']}" for h in hits])
#     if hits and hits[0]["rerank_score"] is not None:
#         chart["rerank prob"] = [sigmoid(h["rerank_score"]) for h in hits]
#     #st.bar_chart(chart, height=240, key=f"bc_{key}")
#     st.bar_chart(chart, height=240)
#     with st.expander("Full text of the chunks sent to the LLM"):
#         for h in hits:
#             if h["in_context"]:
#                 st.markdown(f"**#{h['rank']} - {h['filename']}, page {h['page']}**")
#                 st.text(h["text"])


def render_retrieval(hits, key):
    if not hits:
        st.info("No retrieval results found.")
        return

    st.markdown(
        "**🔎 Retrieval visualisation** - ANN candidates (HNSW) "
        "re-ranked by the cross-encoder; ✅ = sent to the LLM"
    )

    # --------------------------------------------------
    # Retrieval results table
    # --------------------------------------------------
    df = pd.DataFrame([{
        "#": h["rank"],
        "used": "✅" if h["in_context"] else "",
        "file": h["filename"],
        "page": (
            str(h["page"])
            if h["page"] == h["page_end"]
            else f"{h['page']}-{h['page_end']}"
        ),
        "dense cos": round(float(h["dense_score"]), 3),
        "rerank p": (
            None if h["rerank_score"] is None
            else round(sigmoid(float(h["rerank_score"])), 3)
        ),
        "ocr": h["method"] == "ocr",
        "snippet": h["text"][:160].replace("\n", " ") + "...",
    } for h in hits])

    # Prevent mixed integer/string page values
    df["page"] = df["page"].astype(str)

    st.dataframe(
        df,
        hide_index=True,
        width="stretch"
    )

    # --------------------------------------------------
    # Retrieval score visualization
    # --------------------------------------------------
    labels = [
        f"#{h['rank']} {h['filename'][:18]} p{h['page']}"
        for h in hits
    ]

    chart_data = {
        "dense cosine": [
            float(h["dense_score"]) for h in hits
        ]
    }

    if hits[0]["rerank_score"] is not None:
        chart_data["rerank prob"] = [
            sigmoid(float(h["rerank_score"])) for h in hits
        ]

    chart = pd.DataFrame(
        chart_data,
        index=labels
    )

    # Streamlit bar_chart does not accept key in your version
    st.bar_chart(
        chart,
        height=240,
        width="stretch"
    )

    # --------------------------------------------------
    # Full retrieved context
    # --------------------------------------------------
    with st.expander("Full text of the chunks sent to the LLM"):
        for h in hits:
            if h["in_context"]:
                st.markdown(
                    f"**#{h['rank']} - {h['filename']}, "
                    f"page {h['page']}**"
                )
                st.text(h["text"])


def render_sources(sources, key):
    if not sources:
        return
    st.markdown("**📎 Sources**")
    cols = st.columns(min(len(sources), 5))
    for col, s in zip(cols, sources):
        with col:
            pg = f"p.{s['page']}" if s["page"] == s["page_end"] else f"pp.{s['page']}-{s['page_end']}"
            st.caption(f"{'✅ cited' if s['cited'] else '· retrieved'} · {s['id']}")
            st.markdown(f"**{s['filename']}** {pg}")
            img = page_png(s["filename"], s["page"])
            if img:
                st.image(img, width="stretch")


def render_timings(t, key):
    total = t.get("total_ms", 0)
    ok = total <= C.LATENCY_TARGET_MS
    c = st.columns(6)
    c[0].metric("Total", f"{total/1000:.2f}s", "within 5s" if ok else "over 5s", delta_color="normal" if ok else "inverse")
    c[1].metric("Query embed", f"{t.get('embed_ms', 0):.0f} ms")
    c[2].metric("ANN search", f"{t.get('search_ms', 0):.0f} ms")
    c[3].metric("Rerank", f"{t.get('rerank_ms', 0):.0f} ms")
    c[4].metric("LLM 1st token", f"{t.get('llm_ttft_ms', 0):.0f} ms")
    c[5].metric("LLM total", f"{t.get('llm_total_ms', 0):.0f} ms")


# ------------------------------------------------------------------ sidebar
man = load_manifest()
with st.sidebar:
    st.title("📚 PDF RAG Chatbot")
    st.caption(f"Embeddings: `{C.EMBED_MODEL}`  \nVector DB: ChromaDB (HNSW)  \nLLM: `{C.LLM_MODEL}`")
    st.metric("PDFs ingested", len(man))
    st.metric("Pages / chunks", f"{sum(m['pages'] for m in man.values()):,} / {sum(m['chunks'] for m in man.values()):,}")
    st.divider()
    top_k = st.slider("ANN candidates (top-K)", 5, 50, C.TOP_K)
    top_n = st.slider("Chunks sent to LLM (top-N)", 1, 10, C.TOP_N)
    rerank = st.toggle("Cross-encoder rerank", C.USE_RERANK)
    files = st.multiselect("Restrict to PDFs", sorted({m["filename"] for m in man.values()}))
    if st.button("Clear chat"):
        st.session_state.msgs = []
        st.rerun()

tab_chat, tab_ing, tab_corpus = st.tabs(["💬 Chat", "📥 Ingestion pipeline", "📊 Corpus"])

# ------------------------------------------------------------------ chat
with tab_chat:
    st.session_state.setdefault("msgs", [])
    for i, m in enumerate(st.session_state.msgs):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            d = m.get("details")
            if d:
                render_timings(d["timings"], f"h{i}")
                render_sources(d["sources"], f"h{i}")
                with st.expander("Retrieval details"):
                    render_retrieval(d["hits"], f"h{i}")

    q = st.chat_input("Ask a question about your PDFs...")
    if q:
        if not man:
            st.warning("No documents ingested yet - use the **Ingestion pipeline** tab or run `python -m rag.ingest`.")
            st.stop()
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        pipe = get_pipeline()
        with st.chat_message("assistant"):
            ans_box, time_box, src_box = st.empty(), st.container(), st.container()
            viz_box = st.expander("Retrieval details", expanded=True)
            acc, hits, done = "", None, None
            for ev in pipe.ask_stream(q, top_k=top_k, top_n=top_n, rerank=rerank, files=files or None):
                if ev["type"] == "retrieval":
                    hits = ev["hits"]
                    ans_box.markdown(f"_retrieved in {ev['timings']['retrieval_ms']:.0f} ms - generating..._")
                    with viz_box:
                        render_retrieval(hits, f"live{len(st.session_state.msgs)}")
                elif ev["type"] == "token":
                    acc += ev["text"]
                    ans_box.markdown(acc + "▌")
                elif ev["type"] == "done":
                    done = ev
                    ans_box.markdown(ev["answer_cited"])
                    with time_box:
                        render_timings(ev["timings"], "live")
                    with src_box:
                        render_sources(ev["sources"], "live")
                elif ev["type"] == "error":
                    ans_box.error(ev["message"] + "  \nIs your LLM running / API key set? See README → Configure the LLM.")
            if done:
                st.session_state.msgs.append({"role": "assistant", "content": done["answer_cited"],
                                              "details": {"timings": done["timings"], "sources": done["sources"], "hits": hits}})

# ------------------------------------------------------------------ ingestion
with tab_ing:
    st.markdown("### Document ingestion pipeline")
    st.markdown("`PDF → native text / OCR → clean (headers, footers, hyphens) → language detect → "
                f"chunk ({C.CHUNK_TOKENS} tok, {int(C.OVERLAP_RATIO*100)}% overlap) → embed → HNSW index`")
    st.caption("For the big initial load use the CLI (`python -m rag.ingest`); this tab is ideal for adding PDFs live during a demo. "
               "Already-ingested PDFs are skipped (content-hash based).")
    ups = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    c1, c2 = st.columns(2)
    run_up = c1.button("Ingest uploaded files", disabled=not ups)
    run_dir = c2.button(f"Ingest everything in {C.PDF_DIR.name}/")
    if run_up or run_dir:
        from rag.ingest import ingest_pdf
        targets = []
        if run_up:
            for u in ups:
                dest = C.PDF_DIR / u.name
                dest.write_bytes(u.getbuffer())
                targets.append(dest)
        else:
            targets = sorted(C.PDF_DIR.glob("*.pdf"))
        rows = []
        for t in targets:
            st.write(f"**{t.name}**")
            bar, msg = st.progress(0.0), st.empty()
            r = ingest_pdf(t, progress=lambda s, f, m: (bar.progress(min(f, 1.0)), msg.caption(f"{s}: {m}")), workers=C.WORKERS)
            msg.caption("already ingested - skipped" if r["skipped"] else f"done: {r['chunks']} chunks")
            rows.append({"file": t.name, "pages": r["pages"], "ocr pages": r["ocr_pages"], "chunks": r["chunks"],
                         **({} if r["skipped"] else {f"{k} s": v for k, v in r["seconds"].items()})})
            for e in r.get("ocr_errors", []):
                st.warning(f"OCR error: {e} - is Tesseract installed?")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.success("Ingestion finished. Reload the page to refresh the sidebar counters.")

# ------------------------------------------------------------------ corpus
with tab_corpus:
    if man:
        st.dataframe(pd.DataFrame(man.values()), hide_index=True, width="stretch")
    else:
        st.info("Nothing ingested yet.")
