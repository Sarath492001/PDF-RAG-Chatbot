"""Central configuration. Every value can be overridden with an environment variable or .env file."""
import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _b(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
PDF_DIR = Path(os.getenv("PDF_DIR", DATA_DIR / "pdfs"))
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
for _d in (PDF_DIR, PROCESSED_DIR / "pages", PROCESSED_DIR / "chunks", CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Extraction / OCR ----
OCR_ENABLED = _b("OCR_ENABLED", True)
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", 200))
MIN_NATIVE_CHARS = int(os.getenv("MIN_NATIVE_CHARS", 60))   # fewer chars than this -> page treated as scanned
WORKERS = int(os.getenv("WORKERS", max(1, (os.cpu_count() or 2) - 1)))
PAGES_PER_TASK = 10

# ---- Chunking (token counts use the embedding model's own tokenizer) ----
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", 480))          # bge-small max is 512 -> 480 leaves room for special tokens
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", 0.15))     # 10-30 % per the brief

# ---- Embeddings ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
DOC_PREFIX = os.getenv("DOC_PREFIX", "")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 512))
DEVICE = os.getenv("DEVICE") or None                         # cuda | mps | cpu | None = auto

# ---- Vector DB (ChromaDB, HNSW) ----
COLLECTION = ("chunks_" + re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL).strip("_"))[:60]
MANIFEST = PROCESSED_DIR / f"manifest_{COLLECTION}.json"
HNSW_M = int(os.getenv("HNSW_M", 32))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", 200))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", 100))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", 20))                          # ANN candidates
TOP_N = int(os.getenv("TOP_N", 5))                           # passages sent to the LLM
USE_RERANK = _b("USE_RERANK", True)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---- LLM (OpenAI-compatible: Ollama, Groq, vLLM, LM Studio, ...) ----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 400))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", 180))
LATENCY_TARGET_MS = 5000
