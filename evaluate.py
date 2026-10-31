"""Evaluation: retrieval quality (R@k, MRR), latency (p50/p95), citation accuracy, hallucination rate.

  python evaluate.py generate --n 100          # synthetic Q&A set from random chunks  -> eval/eval_set.jsonl
  python evaluate.py run --no-llm              # retrieval metrics only (fast, no LLM needed)
  python evaluate.py run --judge               # + end-to-end latency, citation accuracy, hallucination (LLM-judge)

You can also hand-write eval/eval_set.jsonl lines: {"question": "...", "filename": "x.pdf", "page": 12, "page_end": 12}
Set JUDGE_MODEL (and LLM_* vars) to use a stronger model as judge than the one that generates answers.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from rag import config as C
from rag.generate import NOT_FOUND, RAGPipeline

EVAL_SET = Path("eval/eval_set.jsonl")
KS = (1, 3, 5, 10)


def match(h, g):
    """Page-level hit: same PDF and page ranges overlap."""
    return h["filename"] == g["filename"] and h["page"] <= g["page_end"] and h["page_end"] >= g["page"]


def llm(client, model, prompt, max_tokens=120):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()


def generate(n):
    pipe = RAGPipeline()
    col = pipe.retriever.store.col
    ids = col.get(include=[])["ids"]
    random.Random(42).shuffle(ids)
    out, i = [], 0
    while len(out) < n and i < len(ids):
        batch = col.get(ids=ids[i:i + 50], include=["documents", "metadatas"])
        i += 50
        for doc, m in zip(batch["documents"], batch["metadatas"]):
            if len(out) >= n or len(doc) < 400:
                continue
            q = llm(pipe.client, C.LLM_MODEL,
                    "Write ONE specific question that can be answered using only the passage below. "
                    "Do not mention 'the passage' or 'the text'. Output only the question.\n\nPassage:\n" + doc[:2500])
            out.append({"question": q.strip().strip('"'), "filename": m["filename"], "page": m["page"], "page_end": m["page_end"]})
            print(f"{len(out)}/{n} {q[:90]}")
    EVAL_SET.parent.mkdir(exist_ok=True)
    EVAL_SET.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out))
    print(f"wrote {len(out)} questions -> {EVAL_SET}")


def run(no_llm, judge, limit):
    rows = [json.loads(l) for l in EVAL_SET.read_text().splitlines() if l.strip()][:limit]
    pipe = RAGPipeline()
    pipe.warmup()
    jmodel = os.getenv("JUDGE_MODEL", C.LLM_MODEL)
    recalls = {k: [] for k in KS}
    rr, ret_ms, tot_ms, ftok_ms = [], [], [], []
    cite_ok, cite_invalid, uncited, abstain, halluc, judged = [], [], [], 0, 0, 0
    for n, g in enumerate(rows, 1):
        retrieval, done = None, None
        for ev in pipe.ask_stream(g["question"]):
            if ev["type"] == "retrieval":
                retrieval = ev
                if no_llm:
                    break
            elif ev["type"] == "done":
                done = ev
            elif ev["type"] == "error":
                print("LLM error:", ev["message"])
        hits = retrieval["hits"]
        ret_ms.append(retrieval["timings"]["retrieval_ms"])
        flags = [match(h, g) for h in hits[:max(KS)]]
        for k in KS:
            recalls[k].append(any(flags[:k]))
        rr.append(next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0))
        if done and not no_llm:
            tot_ms.append(done["timings"]["total_ms"])
            ftok_ms.append(done["timings"].get("first_token_ms", done["timings"]["total_ms"]))
            if done["answer"].startswith(NOT_FOUND[:30]):
                abstain += 1
            else:
                ctx = [h for h in hits if h["in_context"]]
                cited = [ctx[i - 1] for i in done["cited_ids"]]
                cite_ok.append(any(match(h, g) for h in cited))
                cite_invalid.append(bool(done["invalid_citations"]))
                uncited.append(not done["cited_ids"])
                if judge:
                    ctx_txt = "\n\n".join(h["text"] for h in ctx)
                    v = llm(pipe.client, jmodel,
                            "You are a strict fact-checker. Is EVERY claim in the ANSWER supported by the CONTEXT? "
                            f"Reply with exactly SUPPORTED or UNSUPPORTED.\n\nCONTEXT:\n{ctx_txt}\n\nANSWER:\n{done['answer']}", 5)
                    judged += 1
                    halluc += v.upper().startswith("UNSUPPORTED")
        if n % 10 == 0:
            print(f"  {n}/{len(rows)}")

    pct = lambda a, p: float(np.percentile(a, p)) if len(a) else None
    rep = {"n_questions": len(rows), "retrieval": {f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()},
           "mrr@10": float(np.mean(rr)), "retrieval_latency_ms": {"p50": pct(ret_ms, 50), "p95": pct(ret_ms, 95)}}
    if tot_ms:
        answered = len(tot_ms) - abstain
        rep["end_to_end_latency_s"] = {"p50": pct(tot_ms, 50) / 1000, "p95": pct(tot_ms, 95) / 1000,
                                       "first_token_p95": pct(ftok_ms, 95) / 1000,
                                       "within_5s_pct": float(np.mean([t <= 5000 for t in tot_ms]))}
        rep["answers"] = {"abstained": abstain, "answered": answered,
                          "citation_accuracy(gold page among cited)": float(np.mean(cite_ok)) if cite_ok else None,
                          "answers_without_any_citation": float(np.mean(uncited)) if uncited else None,
                          "invalid_citation_rate": float(np.mean(cite_invalid)) if cite_invalid else None,
                          "hallucination_rate(LLM-judge)": (halluc / judged) if judged else None}
    Path("eval").mkdir(exist_ok=True)
    Path("eval/report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=100)
    r = sub.add_parser("run"); r.add_argument("--no-llm", action="store_true"); r.add_argument("--judge", action="store_true")
    r.add_argument("--limit", type=int, default=10**9)
    a = ap.parse_args()
    generate(a.n) if a.cmd == "generate" else run(a.no_llm, a.judge, a.limit)
