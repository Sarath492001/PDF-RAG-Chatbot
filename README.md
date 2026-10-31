# PDF RAG Chatbot

A Python application for asking questions about a collection of PDF documents. It extracts text from each page, uses OCR when a page appears to be scanned, creates searchable chunks, and retrieves relevant passages when a question is asked. Answers include references to the source PDF and page number.

The project uses local document processing, BGE embeddings, ChromaDB, and an optional OpenAI-compatible language-model endpoint.

## Features

- Extracts native PDF text with PyMuPDF.
- Uses Tesseract OCR for pages with little extractable text or pages that are mostly images. It does not yet OCR every image region embedded in an otherwise text-rich page.
- Processes pages in parallel during ingestion.
- Removes repeated headers, footers, and page-number lines where detected.
- Detects the language of each page.
- Splits text into token-based chunks and stores page and bounding-box metadata.
- Generates embeddings with `BAAI/bge-small-en-v1.5`.
- Stores vectors in a persistent ChromaDB collection using HNSW.
- Retrieves the nearest passages and can rerank them with a cross-encoder.
- Generates answers through an OpenAI-compatible endpoint, including Ollama.
- Provides a Streamlit interface and a FastAPI service.
- Includes scripts for retrieval and answer evaluation.

## Project structure

```text
rag-chatbot/
├── rag/
│   ├── config.py       # Environment-based settings
│   ├── extract.py      # PDF text extraction, OCR, and cleaning
│   ├── chunking.py     # Token-based chunking and metadata
│   ├── store.py        # Embeddings, ChromaDB, and ingestion manifest
│   ├── ingest.py       # Ingestion pipeline and command-line interface
│   ├── retrieve.py     # Vector retrieval and optional reranking
│   └── generate.py     # Prompt construction, LLM calls, and citations
├── data/
│   ├── pdfs/           # Source PDF files
│   ├── processed/      # Cached page extraction and chunk files
│   └── chroma/         # Persistent vector collection
├── eval/               # Evaluation dataset and reports
├── api.py              # FastAPI endpoints
├── app.py              # Streamlit application
├── evaluate.py         # Evaluation commands
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Requirements

The project is intended for Python 3.11. A compatible PyTorch build is required by Sentence Transformers.

Install Tesseract OCR as a system package:

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS**

```bash
brew install tesseract
```

On Windows, install Tesseract and add its executable to `PATH`.

Check the installation:

```bash
tesseract --version
```

You also need an LLM endpoint. Ollama can run a model locally; a hosted OpenAI-compatible endpoint can also be configured.

## Installation

From the project directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python packages:

```bash
pip install -r requirements.txt
```

Copy the example configuration:

```bash
cp .env.example .env
```

On Windows, copy `.env.example` to `.env` using File Explorer or PowerShell. The application loads values from `.env` when `python-dotenv` is installed.

## Configure the language model

### Option 1: Ollama

Install Ollama, then download a model:

```bash
ollama pull llama3.2:3b
```

The default settings in `.env.example` use Ollama:

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2:3b
```

Make sure the Ollama service is running before starting the application:

```bash
ollama serve
```

The model's speed depends on the available CPU, GPU, memory, and model quantization. The first generation request can take longer while the model is loaded.

### Option 2: Hosted OpenAI-compatible endpoint

The application can also connect to a hosted endpoint, for example Groq, by setting `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in `.env`.

Check the provider's current model list before choosing a model; model identifiers and free-tier limits can change.

**Privacy note:** With a hosted endpoint, the retrieved PDF passages are sent to that provider for answer generation. Use this option only if the document owner permits external processing. For private or restricted documents, use an approved local endpoint.

## Add PDF documents

Copy the source documents into:

```text
data/pdfs/
```

The challenge requires at least 10 PDFs with at least 200 pages each. Check the page count of every file before the demo. The ingestion command processes the PDFs it finds, but it does not enforce the challenge's minimum document count or page count.

It is useful to include both text-based PDFs and at least one scanned PDF so that the OCR path can be demonstrated.

## Ingest the documents

Run the ingestion pipeline from the project root:

```bash
python -m rag.ingest --dir data/pdfs
```

For each PDF, the pipeline:

1. Reads page text and bounding boxes with PyMuPDF.
2. Runs OCR on pages that meet the configured OCR condition.
3. Cleans extracted text and detects the page language.
4. Creates chunks using the embedding model's tokenizer.
5. Generates normalized BGE embeddings.
6. Writes the embeddings and metadata to ChromaDB.

The extraction results and chunks are cached under `data/processed/`. Unchanged PDFs are identified by their content hash and skipped on later runs.

Useful options:

```bash
# Ingest one PDF
python -m rag.ingest --file data/pdfs/example.pdf

# Re-create chunks and embeddings, using cached page extraction
python -m rag.ingest --dir data/pdfs --force

# Re-run extraction and OCR as well
python -m rag.ingest --dir data/pdfs --force --no-cache

# Set the number of extraction worker processes
python -m rag.ingest --dir data/pdfs --workers 4
```

OCR can take considerably longer than native text extraction, especially for high-resolution scanned pages. The extracted pages are cached so OCR does not need to run again unless the cache is removed or `--no-cache` is used.

## Run the application

### Streamlit interface

```bash
streamlit run app.py
```

Open the local address printed by Streamlit, usually:

```text
http://localhost:8501
```

The interface has three tabs:

- **Chat:** ask questions, view the generated answer, source references, retrieved passages, and latency measurements.
- **Ingestion pipeline:** upload PDFs or ingest files already in `data/pdfs/`.
- **Corpus:** view the documents recorded in the ingestion manifest, including page and chunk counts.

The chat sidebar lets you change the number of retrieved candidates, the number of passages sent to the LLM, reranking, and the selected PDF files.

### FastAPI service

Start the API in a separate terminal:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Available endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | Check service and indexed chunk count |
| `GET /documents` | List documents in the ingestion manifest |
| `POST /ask` | Return a complete answer as JSON |
| `POST /ask/stream` | Return newline-delimited JSON events |

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main recommendations?"}'
```

The response includes the answer, source information, retrieved passages, citation information, and timing measurements.

The Streamlit application uses the RAG pipeline directly. If you run the API and Streamlit at the same time, both processes will access the same local ChromaDB files. For this submission, avoid running ingestion from a separate process while another process is actively using the same local ChromaDB collection.


## Docker (optional)

The Dockerfile starts the Streamlit interface. Build the image from the project root:

```bash
docker build -t pdf-rag-chatbot .
```

Run it while mounting the local data directory:

```bash
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v "$PWD/data:/app/data" \
  pdf-rag-chatbot
```

If Ollama is running on the Linux host, add host-gateway access and point the container to the host service:

```bash
docker run --rm -p 8501:8501 \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -v "$PWD/data:/app/data" \
  pdf-rag-chatbot
```

The Docker image does not start the FastAPI service; run that separately if the API is needed. Tesseract is installed in the image, but the LLM service remains external to the container unless you configure and run it separately.

## Retrieval and citations

The current retrieval path is dense vector search in ChromaDB, followed by optional cross-encoder reranking. It does not currently include BM25 or another keyword-search stage.

The retrieved passages are labelled `[S1]`, `[S2]`, and so on in the prompt. The citation resolver maps valid labels back to the PDF filename and page range. The interface distinguishes passages cited in the answer from passages that were retrieved but not cited.

The model is instructed to answer from the retrieved context and to say when the answer is not available. This is a prompt-based safeguard, not a guarantee that every unsupported answer will be rejected. The current implementation also does not independently verify that each cited passage semantically supports the claim next to it. Review citations during evaluation.

## Configuration

The main settings are in `rag/config.py` and can be overridden through environment variables in `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_TOKENS` | `480` | Maximum chunk token budget |
| `OVERLAP_RATIO` | `0.15` | Chunk overlap |
| `OCR_ENABLED` | `true` | Enable OCR fallback |
| `OCR_LANG` | `eng` | Tesseract language |
| `OCR_DPI` | `200` | Page rendering resolution for OCR |
| `MIN_NATIVE_CHARS` | `60` | Native-text threshold for OCR |
| `TOP_K` | `20` | Number of vector candidates |
| `TOP_N` | `5` | Number of passages sent to the LLM |
| `USE_RERANK` | `true` | Enable cross-encoder reranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker |
| `DEVICE` | automatic | `cuda`, `mps`, or `cpu` |
| `HNSW_M` | `32` | HNSW graph parameter |
| `HNSW_EF_CONSTRUCTION` | `200` | HNSW construction parameter |
| `HNSW_EF_SEARCH` | `100` | HNSW search parameter |

BGE-small has a 512-token model input limit. The default chunk budget is 480 tokens to leave room for special tokens. If you change the embedding model or its tokenizer, review `MAX_SEQ_LEN`, `CHUNK_TOKENS`, and the query/document prefixes, then re-ingest the documents into the corresponding collection.

## Evaluation

The evaluation script can create a question set from indexed passages and run retrieval tests.

Generate a set of questions:

```bash
python evaluate.py generate --n 100
```

Run retrieval-only evaluation:

```bash
python evaluate.py run --no-llm
```

Run answer evaluation with an LLM judge:

```bash
python evaluate.py run --judge
```

Results are written to `eval/report.json`.

The report includes Recall@K, MRR, retrieval latency, and, when answer generation is enabled, end-to-end latency and citation-related measurements. The judge-based hallucination measurement is an automated estimate, not a substitute for manual review. By default, the judge uses the configured generation model; set `JUDGE_MODEL` to a stronger model if one is available.

The automatically generated questions are useful for an initial check, but they are not a fully independent benchmark. For a more reliable result, prepare a manually reviewed evaluation set with questions and their expected source PDF and page range.

For the final submission, report the actual results measured on the machine used for the demo. Include the hardware, model, quantization, number of test questions, and whether the measurements were taken after warm-up.

## Latency

The 2–5 second response time is a target, not a fixed guarantee. It depends mainly on the LLM endpoint, hardware, model size, prompt length, reranking, and the number of generated tokens.

The application records query embedding, vector search, reranking, first-token, generation, and total timings. Document extraction, OCR, and embedding generation happen during ingestion and should be reported separately from query-time latency.

Run a representative set of questions and report p50 and p95 latency. Do not use illustrative estimates in place of measured results. The first request may be slower than later requests, particularly when a local LLM has to load into memory.

## Current limitations

- Retrieval is dense-only. BM25 or hybrid retrieval has not been added.
- Follow-up questions are not rewritten using conversation history; each query is retrieved independently.
- The system extracts text from pages but does not interpret figures or diagrams with a vision model.
- Table extraction quality depends on the PDF's text layout.
- OCR accuracy depends on scan quality, resolution, and the installed Tesseract language data.
- Prompt instructions reduce unsupported answers but do not guarantee that hallucinations never occur.
- Citation labels are checked against retrieved passages, but semantic citation support still needs human evaluation.
- The local ChromaDB setup is intended for a single application instance, not concurrent multi-user production use.

## Troubleshooting

| Problem | Things to check |
|---|---|
| `TesseractNotFoundError` | Install Tesseract and check that `tesseract --version` works in the same environment. |
| OCR returns empty text | Check scan quality, OCR language data, and `OCR_DPI`. |
| No documents appear in the UI | Run ingestion and check the manifest and `data/chroma/`. |
| LLM request fails | Check that Ollama or the hosted endpoint is running and that the URL, model name, and API key are correct. |
| Answers do not cite sources | Inspect the retrieved passages and the model output; the model may not have followed the citation instruction. |
| Responses take more than five seconds | Check the LLM generation time first, then test a smaller model, shorter context, or disabled reranking. |
| Results look stale after re-ingestion | Stop other processes using the local ChromaDB directory and restart the application. |

## Before the demo

- Confirm that at least 10 PDFs are present and each has at least 200 pages.
- Include at least one scanned PDF and confirm that OCR extracted useful text.
- Check the ingestion manifest for page and chunk counts.
- Ask questions with known answers and verify the cited PDF pages manually.
- Ask questions that are not covered by the documents and inspect whether the system abstains.
- Run the evaluation script and keep the generated report.
- Measure p50 and p95 latency on the actual demo machine.
