# Modernization log

> A specific, defensible catalogue of what was updated, what was added, and what was deliberately kept. This is the "what changed and why" document — different from `INTERVIEW_GUIDE.md` which is the "how to discuss" document.

---

## The shape of the change

Before this work, the GitHub account had:
- 9 repositories total (this monorepo + 8 separate repos).
- 1 EDA notebook in `Projects` (LA Crime — Jupyter only, no deployment).
- A `langchain-research-assistant` repo using older LangChain (pre-LCEL agentic patterns).
- A `PEGASUS-DialogueSum` repo with full fine-tuning of a 2020-era encoder-decoder.
- Pytorch-Exercises, Langchain-Projects, and Pharma-Connect — learning repos without portfolio polish.
- No deployed artifacts. No structured READMEs. No Dockerfiles. No tests. No deployment scripts.

After this work, the `Projects` repo is a single consolidated monorepo with **four self-contained, deploy-ready projects**, each with structured layout, README, architecture doc, deployment doc, interview cheat-sheet, tests, Docker image, and HF Spaces deploy script.

---

## Decision log — why a monorepo?

| Option | Why I chose against | Why this won |
|---|---|---|
| Update each separate repo in place | Eight repos × four artifacts each = unfocused, hard to navigate, low recruiter signal | One repo = one URL on a CV. Recruiters click one link, see the index. |
| Build new portfolio repos from scratch | Loses provenance and history | Consolidating preserves the original LA Crime notebook in `03-EDA-Dashboards/la-crime-analysis/` as evidence of the project's evolution |
| Keep all old repos *and* build new ones | Looks scattered and uncurated | Old repos can be archived; the README at the root of `Projects/` links to them as "original / archived" for transparency |

---

# Project 01 — Research Assistant (RAG)

**Maps to original repo**: `langchain-research-assistant`

## Library / API updates

| Component | Original | Now | Why the change |
|---|---|---|---|
| Orchestration | LangChain (pre-LCEL, agentic chains) | LangChain 0.3 + LangGraph 0.2 | LangGraph is the recommended pattern for non-linear LLM workflows in 2025+. LCEL replaces older `Chain` classes |
| Vector store | None / FAISS in-process | ChromaDB (persistent, embedded) | Chroma is the de-facto local-first vector DB choice in 2025–2026 |
| Embeddings | Whatever HF default was | `sentence-transformers/all-MiniLM-L6-v2` (explicitly chosen) | CPU-friendly (deploys to HF Spaces free tier), normalized cosine similarity |
| Reranking | None | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) two-stage retrieval | Hybrid retrieve-then-rerank is the production-RAG standard |
| Sparse retrieval | None | BM25 (rank-bm25) alongside dense | Hybrid dense+sparse outperforms either alone on most benchmarks |
| LLM backend | Single hard-coded provider | Pluggable factory: HuggingFace, Groq, Anthropic, OpenAI | Decouples app from vendor; one env var swaps providers |
| Web framework | Bare script / Flask | FastAPI 0.115 + Pydantic v2 | Modern Python API standard; Swagger UI out of the box |
| Frontend | None | Gradio 5 | Deploys to HF Spaces with zero glue code |
| Config | Hard-coded constants | `pydantic-settings.BaseSettings` | Env-var validation at startup, not at first use |
| Logging | Print statements | `structlog` with structured JSON | Standard for production Python |
| Evaluation | None | RAGAS 0.2 suite + golden dataset | Faithfulness, answer-relevancy, context-precision, context-recall |
| Testing | None | Pytest with mocked retriever | Configures + chunking + dedup logic covered |
| CI | None | GitHub Actions (ruff + pytest on push/PR) | |
| Deployment | None | Dockerfile + docker-compose + one-command HF Spaces deploy script | |

## Architectural additions (vs original)

1. **LangGraph state machine** with five named nodes (`rewrite_query, retrieve, grade, generate, reflect`) + one conditional edge for the self-reflection loop. Capped at 1 iteration to bound latency.
2. **Hybrid retrieval** layer at `src/retrieval/retriever.py` — dense from Chroma, sparse from BM25, deduplicated by content, reranked by cross-encoder.
3. **Multi-provider LLM factory** at `src/llm/providers.py` — the only file in the codebase that knows about provider SDKs.
4. **Citation tracking** baked into the prompt format — context blocks are numbered `[1], [2], ...` and the model is instructed to cite by number.
5. **Lazy imports** in `src/retrieval/vector_store.py` and `src/retrieval/reranker.py` so unit tests run without the heavy ML deps installed.

## Things deliberately kept simple

- Single reflection iteration (not unbounded loops)
- In-process BM25 (rebuilt on ingest, not persisted)
- No streaming responses yet (could be added easily; not core to demo)

---

# Project 02 — Dialogue Summarization

**Maps to original repo**: `PEGASUS-DialogueSum`

## The big migration

| | Original PEGASUS work | This project |
|---|---|---|
| **Base model** | PEGASUS-large (568M, encoder-decoder, 2020) | Llama-3.2-3B-Instruct (decoder-only, 2024) |
| **Fine-tuning method** | Full fine-tune | LoRA (PEFT, r=16, α=32 on q/k/v/o) |
| **Quantization** | None | 4-bit NF4 (QLoRA) |
| **Trainer** | `transformers.Trainer` | `trl.SFTTrainer` with `assistant_only_loss=True` |
| **Data format** | Raw (dialogue, summary) pairs | Llama-3 chat-template messages |
| **Evaluation** | ROUGE only | ROUGE-1/2/L (rouge-score) + BERTScore-F1 |
| **Artifact** | Local checkpoint | LoRA adapter pushed to HF Hub + Gradio Space |
| **Required hardware** | Multi-GB VRAM for full fine-tune | Fits free Colab T4 (~10-12 GB peak) |
| **Adapter size** | Multi-GB | ~30 MB |

## Library updates

| Component | Now |
|---|---|
| transformers | 4.46+ |
| peft | 0.13+ |
| trl | 0.12+ |
| accelerate | 1.1+ |
| bitsandbytes | 0.44+ |

## Architectural additions

1. **Chat-template formatter** (`src/data.py:_format_row`) — turns `(dialogue, summary)` rows into Llama-3 chat messages so the fine-tune aligns with the model's instruction prior.
2. **QLoRA loader** (`src/model.py:load_for_training`) — 4-bit NF4 + `prepare_model_for_kbit_training` + LoRA injection.
3. **SFTTrainer config** with `assistant_only_loss=True` — loss only on assistant tokens, not on prompt.
4. **Inference paths** with optional adapter merge (`merge_and_unload`) for faster generation.
5. **Auto-generated model card** in `scripts/push_to_hub.py` — includes full training recipe and a working `transformers` snippet.
6. **Gradio app with zero-shot fallback** — runnable even before adapter is trained.

---

# Project 03 — LA Crime Dashboard

**Maps to original repo**: original `Projects/EDA/Los Anglos Crime Analysis/`

## The transformation

| | Original notebook | This dashboard |
|---|---|---|
| **Format** | Single Jupyter notebook with three fixed questions | Streamlit multi-page app, four analytical views |
| **Interactivity** | None (re-run cells to change anything) | Sidebar filters drive every chart live |
| **Charts** | Matplotlib + seaborn static | Plotly interactive (hover, zoom, export) |
| **Deployment** | None (notebook to read) | Dockerfile + HF Spaces deploy script |
| **Testability** | Zero (mixed code + analysis) | Pure function `add_features()` + 7 unit tests |
| **Architecture** | Procedural cells | `src/` package: `data.py`, `filters.py`, `viz.py` |

## Library / framework additions

- Streamlit 1.40 (multi-page convention via `pages/`)
- Plotly Express 5.24 + Graph Objects
- `@st.cache_data` for the 27MB CSV parse (massive UX win)

## Architectural additions

1. **Separation of loading + feature engineering** — `load_raw()` and `add_features()` are independent so storage can be swapped (CSV → Parquet → SQL) without touching feature logic.
2. **Shared sidebar filters** — `src/filters.py:sidebar_filters` used by every page, ensures consistent UX.
3. **Reusable chart library** — `src/viz.py` returns `plotly.graph_objects.Figure` objects so pages can compose and style.
4. **Multi-page layout**: Overview, Geography, Time Patterns, Demographics.
5. **Ethical framing** on the Demographics page — explicit warning that this is reported-victim data, not predicted-offender data, plus the reporting-bias caveat.

## Things preserved
- Original `notebook.ipynb` is intentionally kept alongside the dashboard as evidence of the project's evolution.
- The original `crimes.csv` is unchanged.

---

# Project 04 — Banking Intent Classifier

**Maps to original repo**: derived from `Pytorch-Exercises-and-Projects` (the "deep learning" slot in the portfolio)

This project doesn't have a 1:1 predecessor in the original repos — it was deliberately built to round out the portfolio with **discriminative transformer fine-tuning**, in contrast to Project 2's generative LoRA work.

## Architecture choices and rationale

| Choice | Why |
|---|---|
| **ModernBERT-base** (not BERT-base or DeBERTa-v3) | December 2024 release from Answer.AI; RoPE + GeGLU + alternating local/global attention; trained on ~2T tokens. Drop-in replacement for any BERT-family encoder, materially better representations. |
| **Banking77** (not AG News, DBpedia, etc.) | 77 fine-grained classes = realistic intent-classification challenge. Standard NLU benchmark with published baselines. |
| **Full fine-tune** (not LoRA) | 149M params + 13k examples is the right regime for full fine-tune. Deliberately contrasted with Project 2 to show method-selection awareness. |
| **transformers.Trainer** (not TRL) | Discriminative classification, not instruction-tuning. Trainer is the right tool. |
| **Stratified val split out of train** | Banking77 ships train/test only. Test is held out for honest final reporting. |
| **Macro F1 for selection** | 77 classes with imbalanced sizes — plain accuracy hides poor performance on rare classes. |
| **Label smoothing 0.1** | Regularizer that pairs naturally with temperature scaling; small accuracy bump, better calibration. |
| **Temperature scaling for calibration** | Modern neural classifiers are overconfident. T-scaling fits one scalar to align confidence with accuracy. Reported ECE before/after. |
| **Top-5 in the demo, not top-1** | Semantically near-duplicate intents (e.g., `card_payment_fee_charged` vs `cash_withdrawal_charge`) are common error modes. A real system would route to human / clarification when top-2 are close. |

## Library / framework choices

- transformers 4.48+ (for ModernBERT support)
- accelerate, bitsandbytes (training optional extras)
- scikit-learn (eval metrics)
- gradio (demo)
- pure-numpy temperature scaling (no torch needed for the calibration module)

## Tests
- Calibration is fully unit-tested with synthetic overconfident logits:
  - `fit_temperature` returns T > 1 for overconfident inputs ✓
  - Argmax is preserved under temperature scaling ✓
  - ECE strictly decreases after fitting T ✓
  - `apply_temperature` produces a valid probability distribution ✓

---

# Cross-cutting conventions applied to every project

These are the patterns made consistent across all four projects:

## Repository layout (every project)

```
NN-Category/<project>/
├── README.md
├── pyproject.toml          # PEP 621 packaging
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── app.py                  # entry point (or src/__main__.py)
├── src/                    # library code
├── tests/                  # pytest suite
├── scripts/
│   └── deploy_hf_space.sh  # one-command deploy
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── INTERVIEW_NOTES.md
├── notebooks/              # if any
└── .huggingface/
    └── SPACE_README.md     # HF Spaces metadata header
```

## Configuration pattern (every project)

- `src/config.py` with `pydantic_settings.BaseSettings`
- All knobs documented in `.env.example`
- Default values are the values used in the canonical training/deployment
- Type-checked at startup (fail fast on misconfig)

## Deployment pattern (every project)

1. `Dockerfile` makes the project self-contained
2. `.huggingface/SPACE_README.md` carries the HF metadata header
3. `scripts/deploy_hf_space.sh` script:
   - Creates the Space via `huggingface-cli`
   - Swaps `README.md` for the Space-formatted header
   - Uploads code with `huggingface-cli upload`
   - Pushes `.env` values as Space Secrets
   - Restores the GitHub `README.md`

## Documentation pattern (every project)

Three doc files per project:
- `ARCHITECTURE.md` — system diagram + why each component
- `DEPLOYMENT.md` — exact commands for HF Spaces and Docker
- `INTERVIEW_NOTES.md` — question-by-question cheat-sheet

## Testing pattern (every project)

- `tests/` directory with `pytest` configured in `pyproject.toml`
- Tests cover **pure functions** (config loading, feature engineering, calibration, chunking) — the bits that don't need GPU or network
- Lazy imports in modules that depend on heavy ML libs, so tests run without them installed

---

# What is *not* in this portfolio (and why I'd add it next)

1. **MLOps end-to-end project** — MLflow tracking, DVC for data versioning, GitHub Actions doing training in CI, monitoring with Prometheus/Grafana. This is the strongest signal for an *ML Platform* / *ML Engineer at scale* role. If I had another 2–3 days, this would be Project 05.

2. **Multi-modal RAG** — A version of Project 01 that ingests PDFs with figures intact, using a vision-language model for image understanding. Academic papers lose ~30% of their information when stripped to text.

3. **Online evaluation** — All four projects evaluate offline. A production system needs online A/B testing infrastructure with user feedback loops.

4. **Geocoding for the LA Crime dashboard** — The raw data has street addresses; turning them into lat/lon and adding a real choropleth would dramatically upgrade Project 03.

5. **Cross-lingual extension of Banking77** — `MASSIVE` (Amazon) or `XNLI` (Facebook) would let Project 04 demonstrate cross-lingual transfer learning.

---

# Original repos referenced in this monorepo

The root `README.md` links to the original GitHub repos under "Original / archived repos". Those repos can be:
- Left alone (the monorepo is the canonical version going forward)
- Archived (preferred — signals "this is the current work")
- Pinned-with-link-to-monorepo on the GitHub profile

If asked in an interview, the answer is: "I consolidated everything into a single portfolio repo so recruiters see one curated index. The originals are still there as evidence of the journey."
