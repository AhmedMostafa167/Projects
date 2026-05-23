# Research Assistant — Production-Grade RAG with LangGraph

> An end-to-end Retrieval-Augmented Generation (RAG) system that ingests academic papers from ArXiv and the open web, builds a hybrid vector index, and answers research questions with grounded citations. Built with LangChain 0.3 + LangGraph, deployable to Hugging Face Spaces in one command.

[![Deploy on Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-sm.svg)](https://huggingface.co/spaces)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)

---

## What this project demonstrates

| Skill | How it's shown here |
|---|---|
| **Modern LLM orchestration** | LangGraph state machine (not just an LCEL chain) — query planning → retrieval → grading → generation → reflection |
| **Production RAG patterns** | Hybrid retrieval (dense + BM25), cross-encoder reranking, parent-document chunking, citation tracking |
| **Multi-provider LLM design** | Pluggable backend: Hugging Face Inference, Groq, Anthropic, OpenAI — swap with one env var |
| **API + UI separation** | FastAPI service + Gradio frontend, both Dockerized |
| **Deployment** | Hugging Face Spaces (Docker SDK), with a one-command deploy script |
| **Evaluation** | RAGAS-based eval suite (faithfulness, context precision, answer relevancy) |
| **Observability** | Optional LangSmith tracing, structured logging, request IDs |

---

## Architecture

```
                    ┌─────────────────────────┐
   user question →  │   FastAPI / Gradio UI   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LangGraph Workflow     │
                    │   (state machine)        │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌──────────────┐   ┌──────────────┐  ┌──────────────┐
      │ Query        │   │ Hybrid       │  │ Generation   │
      │ Decomposer   │──▶│ Retriever    │─▶│ + Reflection │
      └──────────────┘   │ • Chroma     │  └──────────────┘
                         │ • BM25       │          │
                         │ • Reranker   │          ▼
                         └──────┬───────┘   ┌──────────────┐
                                │           │  Citations + │
                                ▼           │  Answer      │
                         ┌──────────────┐   └──────────────┘
                         │ Ingestion    │
                         │ • ArXiv API  │
                         │ • Web search │
                         │ • PDF        │
                         └──────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system design.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/AhmedMostafa167/Projects.git
cd Projects/01-LLM-RAG/research-assistant
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set HUGGINGFACEHUB_API_TOKEN (free)
```

The app supports multiple LLM providers. The default uses the Hugging Face Inference API (free tier). To use Groq (fast, free tier) or Anthropic/OpenAI, set the corresponding env var and change `LLM_PROVIDER`.

### 3. Run the Gradio app

```bash
python app.py
# → http://localhost:7860
```

### 4. Or run the FastAPI service

```bash
uvicorn api:app --reload
# → http://localhost:8000/docs (Swagger UI)
```

### 5. Run with Docker

```bash
docker compose up --build
```

---

## Deployment

### Hugging Face Spaces (one command)

```bash
bash scripts/deploy_hf_space.sh <your-hf-username> <space-name>
```

This script:
1. Creates the HF Space (Docker SDK) via the HF API
2. Adds it as a git remote
3. Pushes the current code
4. Configures the required secrets from your local `.env`

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for other targets (Render, Railway, AWS).

---

## Evaluation

Run the RAGAS-based eval suite on a golden dataset:

```bash
python -m src.evaluation.run_eval --dataset data/eval_questions.jsonl
```

Outputs faithfulness, answer relevancy, context precision, and context recall scores. Results are logged to `eval_results/` and (optionally) LangSmith.

---

## Project layout

```
research-assistant/
├── app.py                    # Gradio frontend (entry for HF Spaces)
├── api.py                    # FastAPI service
├── src/
│   ├── config.py             # Pydantic settings
│   ├── ingestion/            # ArXiv, web, PDF loaders + chunking
│   ├── retrieval/            # Hybrid retriever, vector store, reranker
│   ├── llm/                  # Multi-provider LLM factory + prompts
│   ├── graph/                # LangGraph workflow definition
│   ├── evaluation/           # RAGAS eval runner
│   └── utils/                # Logging, tracing
├── tests/                    # Pytest unit + integration tests
├── notebooks/                # Exploration notebook (dev journey)
├── docs/
│   ├── ARCHITECTURE.md       # System design deep-dive
│   └── DEPLOYMENT.md         # All deploy targets
├── scripts/
│   └── deploy_hf_space.sh    # One-command HF Spaces deploy
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Tech stack

- **LLM orchestration**: LangChain 0.3, LangGraph 0.2
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly, 384-dim)
- **Vector store**: ChromaDB (persistent, embedded)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM providers**: HF Inference, Groq, Anthropic, OpenAI (pluggable)
- **API**: FastAPI 0.115 + Pydantic v2
- **UI**: Gradio 5
- **Eval**: RAGAS 0.2
- **Observability**: LangSmith (optional), structlog
- **Packaging**: `pyproject.toml`, ruff, mypy, pytest

---

## Design highlights

- **Why LangGraph over plain LCEL chains** — the flow isn't linear; retrieval quality decides whether we generate or fetch more, which is naturally a state machine.
- **Why hybrid retrieval + cross-encoder reranking** — dense embeddings catch semantic matches, BM25 catches exact-keyword matches, the cross-encoder gives precision at the top of the list.
- **Why a pluggable LLM factory** — the rest of the code depends only on `BaseChatModel`, so swapping HF Inference for Groq, Anthropic, or OpenAI is one env var.
- **How citation tracking works** — the answer prompt numbers context blocks `[1], [2], ...` and the model is instructed to cite by number; sources are returned alongside the answer.
- **Scaling beyond ~1M docs** — move the vector store to Qdrant/Weaviate, BM25 to OpenSearch, add a retrieval cache and a queue-based ingestion path.
