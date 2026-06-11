# Research Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that answers research questions grounded in evidence from ArXiv papers and the open web.

[![Demo on Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/Ahmed167/research_assistant)

---

## What it does

1. **Ingest** a research topic — fetches papers from ArXiv and results from the open web, chunks them, and indexes them into a local vector store.
2. **Ask** grounded questions — retrieves the most relevant evidence, grades it, generates a cited answer, and self-reflects to fill gaps if needed.

---

## Architecture

```
User
 │
 ├── ingest(topic) ──→ ArXiv + DuckDuckGo → chunk → Chroma + BM25 index
 │
 └── ask(question) ──→ LangGraph workflow
                            │
                            ├── rewrite_query   (3 search queries via structured output)
                            ├── retrieve        (hybrid: Chroma dense + BM25 sparse → cross-encoder rerank)
                            ├── grade           (per-document relevance filter)
                            ├── generate        (cited answer from graded documents)
                            └── reflect         (sufficient? → done | gap found? → retrieve again)
```

---

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Retrieval strategy | Hybrid (dense + BM25) + reranking | Dense misses rare keywords; BM25 misses semantic paraphrases; cross-encoder reranks both precisely |
| Vector store | Chroma (embedded) | Zero ops, persists to disk, works on HF Spaces; swap to Qdrant at scale |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) | CPU-friendly, no GPU required |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Two-stage retrieval: fast bi-encoder for candidates, precise cross-encoder for final ranking |
| LLM backend | Pluggable factory (HF / Groq / Anthropic / OpenAI) | Single env var swap, zero code changes |
| Ingestion sources | ArXiv + DuckDuckGo directly (no LangChain wrappers) | Full control over `Document` metadata for citation tracking |
| Model loading | `@lru_cache(maxsize=1)` | Lazy loading — models load on first call, not at import time |

---

## Tech stack

- **Orchestration** — LangGraph
- **LLM** — Groq (Llama 3.3 70B) / HuggingFace / Anthropic / OpenAI
- **Retrieval** — LangChain Chroma + BM25 + sentence-transformers cross-encoder
- **Ingestion** — `arxiv`, `duckduckgo-search`
- **API** — FastAPI
- **UI** — Gradio
- **Deployment** — Docker, Hugging Face Spaces

---

## Quickstart

### Local (Gradio UI)

```bash
git clone https://github.com/AhmedMostafa167/Projects
cd Projects/01-LLM-RAG/research-assistant
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python app.py
# Open http://localhost:7860
```

### Local (FastAPI)

```bash
uvicorn api:app --reload
# Swagger UI at http://localhost:8000/docs
```

### Docker

```bash
docker compose up
# Gradio → http://localhost:7860
# FastAPI → http://localhost:8000/docs
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `groq`, `huggingface`, `anthropic`, or `openai` |
| `GROQ_API_KEY` | If using Groq | Groq API key |
| `HUGGINGFACEHUB_API_TOKEN` | If using HF | HuggingFace token |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |

---

## Project structure

```
research-assistant/
├── src/
│   ├── config.py          # Pydantic settings — all env vars validated at startup
│   ├── ingestion/         # ArXiv + DuckDuckGo loaders, chunking
│   ├── retrieval/         # Chroma vector store, BM25, hybrid retriever, reranker
│   ├── llm/               # Provider factory + prompt templates
│   ├── graph/             # LangGraph workflow
│   ├── utils/             # Structured logging 
│   └── pipeline.py        # High-level ingest() / ask() API
├── api.py                 # FastAPI service
├── app.py                 # Gradio UI
├── Dockerfile
└── docker-compose.yml
```
