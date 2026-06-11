# Architecture

## Overview

This project is a Retrieval-Augmented Generation (RAG) system. Given a research question, it (1) retrieves relevant evidence from a corpus of ArXiv abstracts and web search results, (2) reasons over the evidence with an LLM, and (3) returns an answer with inline citations.

The novel parts compared to a tutorial RAG are:
1. **Hybrid retrieval** — dense embeddings + BM25 in parallel, merged and reranked.
2. **LangGraph workflow** — query rewriting, evidence grading, generation, and a reflection loop, modeled as a state machine.
3. **Pluggable LLM backend** — one factory, four providers.

## Component map

| Layer | Module | Responsibility |
|---|---|---|
| Config | `src/config.py` | One Pydantic settings object; all env vars validated at startup |
| Ingestion | `src/ingestion/` | Load from ArXiv & DuckDuckGo, chunk with recursive splitting |
| Retrieval | `src/retrieval/` | Chroma (dense) + BM25 (sparse) + cross-encoder rerank |
| LLM | `src/llm/` | Provider factory + prompt templates |
| Orchestration | `src/graph/workflow.py` | LangGraph state machine |
| Pipeline | `src/pipeline.py` | High-level `ingest()` / `ask()` API |
| Interfaces | `api.py`, `app.py` | FastAPI service, Gradio UI |
| Evaluation | `src/evaluation/` | RAGAS metrics runner |

## Data flow

```
                       User
                        │ question
                        ▼
                ┌───────────────────┐
                │ Gradio / FastAPI  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ ResearchPipeline  │  ◀── singleton, holds the Chroma store
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ LangGraph         │
                └─────────┬─────────┘
                          │
   ┌──────────────────────┼──────────────────────┐
   ▼                      ▼                      ▼
rewrite_query  →   retrieve   →   grade   →   generate   →   reflect
                          ▲                                       │
                          └───────────────────────────────────────┘
                                       (one loop allowed)
```

## LangGraph nodes — what each one does

| Node | Purpose | Why it exists |
|---|---|---|
| `rewrite_query` | LLM rewrites the user's question into 2–3 search queries | Improves recall — academic search benefits from keyword variation that users don't naturally produce |
| `retrieve` | Hybrid retrieval over the corpus | The core retrieval step |
| `grade` | Per-document LLM call: is this relevant? | Filters out near-misses that the retriever still ranks high; cheap because we use the small/free LLM here |
| `generate` | LLM produces the answer with inline `[N]` citations | The synthesis step |
| `reflect` | LLM reads its own draft, decides "ok" or "go fetch X" | Lets the system self-correct on questions where the first retrieval missed key evidence |

The reflection loop is capped at **one iteration** (`MAX_REFLECTION_LOOPS = 1`) to bound latency and cost — important on free-tier inference where each call can take seconds.

## Retrieval strategy

**Why hybrid?**
Pure dense retrieval misses rare-keyword matches (e.g., specific algorithm names, model identifiers). Pure BM25 misses semantic paraphrases. Combining them and deduplicating consistently produces a higher-recall candidate set, which the cross-encoder then reorders for precision.

**Pipeline:**
1. Embed the query and the corpus chunks with `all-MiniLM-L6-v2` (384-dim, CPU-friendly).
2. Dense top-10 from Chroma (cosine similarity).
3. Sparse top-10 from BM25 over the same chunks.
4. Merge, dedupe by the first 200 chars of `page_content`.
5. Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each (query, chunk) pair.
6. Return top-5.

## Why pluggable LLM providers?

In development we use the **Hugging Face Inference API** (free tier, no credit card). For higher quality or production load we'd switch to **Groq** (fast Llama 3 inference, generous free tier), **Anthropic** (Claude Haiku for cost-effective quality), or **OpenAI**. The factory in `src/llm/providers.py` is the only place that knows about provider SDKs — everything downstream depends on LangChain's `BaseChatModel` interface.

This is a deliberate design choice: it means swapping providers is a single env var change, not a refactor.

## Trade-offs and limitations

| Trade-off | Choice made | Why | What I'd change at scale |
|---|---|---|---|
| Vector store | Chroma (embedded) | Zero ops, persists to disk, works on HF Spaces | Qdrant or Weaviate for multi-tenant, sharded deployments |
| Embedding dim | 384 (MiniLM) | Fits on free CPU | bge-large (1024) on GPU for higher recall |
| Chunk size | 800 chars | Good middle ground for abstracts | Adaptive sizing per source type |
| BM25 in memory | In-process, rebuilt on ingest | Simple, no extra service | OpenSearch / Elasticsearch for persistence and scale |
| Reflection loops | Capped at 1 | Latency budget | Adaptive: stop when answer "stable" |
| Evaluation | Offline RAGAS | Standard, reproducible | Online A/B + LangSmith feedback loop |

## What runs where

- **Local dev**: `python app.py` (Gradio on :7860) or `uvicorn api:app` (FastAPI on :8000)
- **Docker**: `docker compose up` — both services with persisted Chroma + HF cache volumes
- **HF Spaces**: `bash scripts/deploy_hf_space.sh <user> <space>` — Docker SDK Space, app on :7860
