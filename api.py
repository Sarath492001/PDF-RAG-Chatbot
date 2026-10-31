"""REST API.   uvicorn api:app --port 8000
POST /ask          -> full JSON answer (answer, sources, timings, retrieved chunks)
POST /ask/stream   -> newline-delimited JSON events (retrieval, token..., done)
GET  /documents    -> ingested PDFs      GET /health
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.generate import RAGPipeline
from rag.store import get_store, load_manifest

state = {}


@asynccontextmanager
async def lifespan(app):
    state["pipe"] = RAGPipeline()
    state["pipe"].warmup()          # load models + warm the HNSW index before the first user query
    yield


app = FastAPI(title="PDF RAG Chatbot", lifespan=lifespan)


class AskReq(BaseModel):
    question: str = Field(min_length=2)
    top_k: int | None = None
    top_n: int | None = None
    rerank: bool | None = None
    files: list[str] | None = None


def _kw(r: AskReq):
    return dict(top_k=r.top_k, top_n=r.top_n, rerank=r.rerank, files=r.files)


@app.post("/ask")
def ask(r: AskReq):
    out = state["pipe"].ask(r.question, **_kw(r))
    if out.get("type") == "error":
        raise HTTPException(502, out["message"])
    return out


@app.post("/ask/stream")
def ask_stream(r: AskReq):
    def gen():
        for ev in state["pipe"].ask_stream(r.question, **_kw(r)):
            yield json.dumps(ev, ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/documents")
def documents():
    return list(load_manifest().values())


@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_store().count()}
