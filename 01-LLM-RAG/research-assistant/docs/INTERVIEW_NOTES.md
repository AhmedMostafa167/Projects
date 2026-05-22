# Interview talking points — Research Assistant

This document is your cheat sheet for discussing the project in interviews. Each section starts with a likely question, gives a 30-second answer (what to say out loud), and then the deeper context (what to know in case they probe).

---

## 1. "Walk me through the project."

**30s answer:**
> It's a Retrieval-Augmented Generation system that answers research questions over ArXiv and the open web. The user gives a topic, the system ingests papers and indexes them in a Chroma vector store with BM25 alongside, and then for each question it runs a LangGraph workflow that rewrites the query, retrieves with hybrid search + cross-encoder reranking, grades the evidence, generates an answer with inline citations, and self-reflects to optionally fetch more. The LLM backend is pluggable — Hugging Face Inference by default, with Groq, Anthropic, and OpenAI as alternatives. It's deployable to a Hugging Face Space with one command.

**If they ask for the demo:** point to the live HF Space (you'll have the URL after deploying).

---

## 2. "Why LangGraph instead of a plain LangChain chain?"

**30s answer:**
> A research-answering task isn't strictly linear. Depending on what the retriever returns, I sometimes want to rewrite the query or fetch more documents — that's a graph, not a pipeline. LangGraph models it as an explicit state machine with named nodes and conditional edges, which makes the flow easier to reason about and modify than nested LCEL chains. It also has native streaming of intermediate state, which is useful for a UX that shows what the agent is "thinking."

**Deeper context:**
- LCEL (LangChain Expression Language) is great for linear `prompt | llm | parser` flows.
- For anything with loops, branches, or human-in-the-loop, LangGraph is the recommended pattern from LangChain themselves.
- The graph in this project: `rewrite_query → retrieve → grade → generate → reflect → (END or back to retrieve)`.

---

## 3. "Why hybrid retrieval?"

**30s answer:**
> Dense and sparse retrieval fail in different places. Dense embeddings catch semantic matches — "transformer architecture" finds "self-attention mechanism" — but they can miss exact keyword matches like rare model identifiers or specific algorithm names. BM25 catches those. I run both in parallel, dedupe by content, then re-rank with a cross-encoder. That two-stage retrieve-then-rerank pattern is standard in production RAG because the cross-encoder is too slow to score the whole corpus, but it's much more accurate than a bi-encoder for the top-K.

**Deeper context:**
- Bi-encoder = independent embeddings for query and document → fast, less accurate.
- Cross-encoder = joint encoding of (query, document) → slow, much more accurate.
- Pattern: bi-encoder for recall over the corpus, cross-encoder for precision over the top-K.

---

## 4. "How do you handle hallucinations?"

**30s answer:**
> Three layers. First, the answer prompt explicitly tells the model to use only the provided context and to say so when it's insufficient. Second, every answer cites its sources inline as `[1]`, `[2]`, matching the numbered context blocks the model saw, so a reader can verify. Third, I evaluate with RAGAS — specifically faithfulness, which checks whether each claim in the answer is supported by the retrieved context. That gives me a measurable handle on hallucination rate, not just a hope.

**Deeper context:**
- RAGAS metrics used: faithfulness, answer relevancy, context precision, context recall.
- The eval set is small (5 questions) but designed to be extended.
- Citation tracking is implemented in `src/graph/workflow.py:_format_context` — sources are numbered in the prompt and the model is instructed to use those numbers.

---

## 5. "Why this embedding model? Why this chunk size?"

**30s answer:**
> `all-MiniLM-L6-v2` is 384-dim and ~80MB — it runs comfortably on the Hugging Face Spaces free tier (CPU only), which was a hard constraint. For higher quality at the cost of memory I'd swap to `bge-large-en-v1.5` (1024-dim). Chunk size of 800 characters with 120 overlap is a balance: small enough that multiple chunks fit in the LLM's context window, large enough to preserve local context, and the overlap means we don't lose facts that span chunk boundaries.

**Deeper context:**
- MTEB leaderboard for picking embedding models.
- Chunking is a sensitive hyperparameter — production systems often tune it per source type (e.g. different settings for code vs. prose).
- Alternatives to recursive character splitting: semantic chunking, sentence-window, parent-document.

---

## 6. "How would you scale this to 1M+ documents?"

**30s answer:**
> Four changes. (1) Move the vector store to a managed service — Qdrant Cloud or Weaviate — because Chroma's embedded mode doesn't shard. (2) Move BM25 to OpenSearch or Elasticsearch so it persists and supports updates without rebuilding the in-memory index. (3) Use a fleet of small embedding workers behind a queue for ingestion. (4) Add a caching layer for retrieval results because question popularity follows a long tail. The application code stays mostly the same — the abstractions are at the LangChain interface level, which makes the swap mechanical.

---

## 7. "What's the multi-provider LLM design about?"

**30s answer:**
> The rest of the codebase depends only on LangChain's `BaseChatModel` interface. The factory in `src/llm/providers.py` is the only file that imports provider-specific SDKs. Switching from HF Inference to Groq is a single env var change — `LLM_PROVIDER=groq`. The point is to decouple the application from the LLM vendor: a research project today, a paid model in production, a local model for offline eval, all without code changes.

---

## 8. "What was the hardest part?"

**Honest answers you can pick from depending on what you actually struggled with — pick one and tell the story:**
- "Getting the LangGraph state types right — the `Annotated[list, _append]` accumulator pattern wasn't obvious from the docs."
- "Tuning the reflection loop — it would spiral if I didn't cap iterations, but capping too aggressively meant it never helped."
- "Cold-start performance on HF Spaces — first request had to download embedding + reranker models, which broke the free-tier startup timeout. Fixed by pre-loading in the Dockerfile and using volume caches."

---

## 9. "What would you build differently next time?"

**30s answer:**
> Two things. First, I'd start with the eval set, not the system. Building the RAG and then evaluating it meant I had no way to compare design choices objectively until the end. Second, I'd add LangSmith tracing from day one — debugging the graph by reading logs was painful, and LangSmith's trace view shows exactly what each node received and returned.

---

## 10. "What does the project not do that a production system would?"

**Honest list — you'll get points for self-awareness:**
- No auth on the API
- No rate limiting (relies on provider's)
- Single-tenant — one vector store
- No incremental ingestion / dedupe (re-ingesting the same topic doubles the chunks)
- No PII/safety filtering on outputs
- Evaluation is offline only; no online feedback loop
- No multi-modal (papers with figures lose information)

---

## Quick facts to memorize

- **LangChain** version: 0.3.x
- **LangGraph** version: 0.2.x
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, normalized cosine)
- **Reranker**: `ms-marco-MiniLM-L-6-v2` (cross-encoder)
- **Vector store**: Chroma (persistent, embedded)
- **Sparse retriever**: BM25 (Okapi)
- **Default LLM**: `Llama-3.2-3B-Instruct` via HF Inference
- **Chunk size / overlap**: 800 / 120 characters
- **Top-K**: 10 dense + 10 BM25 → rerank to 5
- **Reflection loop cap**: 1

---

## If they ask "show me one specific code snippet"

Go to `src/graph/workflow.py` and point at:
- The `ResearchState` TypedDict with the `Annotated[list, _append]` accumulator → "this is how LangGraph lets multiple nodes contribute to the same list field safely."
- The conditional edge `workflow.add_conditional_edges("reflect", should_continue, ...)` → "this is what makes it a graph and not a chain."
