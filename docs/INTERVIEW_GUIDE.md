# Portfolio Interview Guide

> The single document to study before NLP/ML engineering interviews. Each project gets a one-page summary; the second half is cross-project material — things interviewers will ask that touch multiple projects.

---

## How to use this document

1. **Read end-to-end once** to absorb the narrative across projects.
2. **For each project**, also read its own `docs/INTERVIEW_NOTES.md` — that has the question-by-question deep dive.
3. **The "Cross-project questions" section** is where interviews actually go — you usually get asked something that spans the portfolio, not a deep technical drill on one piece.
4. **Practice the one-sentence pitches** out loud. If you can't say each project in one sentence, you don't yet own it.

---

## The portfolio in one paragraph

> I built a 4-project monorepo that covers the range of NLP/ML engineering work: a production-grade RAG research assistant with LangGraph; a generative fine-tune of Llama-3.2-3B with QLoRA on SAMSum; a Streamlit dashboard turning raw LAPD data into an interactive analytics tool; and a discriminative fine-tune of ModernBERT on Banking77 with calibrated confidence outputs. Every project has a structured src/ layout, unit tests, Docker, and a deploy script to Hugging Face Spaces.

---

## One-sentence pitches (memorize these)

| Project | One sentence |
|---|---|
| **01 — Research Assistant** | A production-grade RAG system over ArXiv + open web, built as a LangGraph state machine with hybrid retrieval + cross-encoder reranking, with a pluggable multi-provider LLM backend. |
| **02 — Dialogue Summarization** | A QLoRA fine-tune of Llama-3.2-3B-Instruct on SAMSum, using TRL's SFTTrainer with assistant-only loss masking, modernizing my earlier PEGASUS work with parameter-efficient methods. |
| **03 — LA Crime Dashboard** | A multi-page Streamlit + Plotly dashboard that turned a one-off exploratory notebook into a stakeholder-usable tool with shared filters and four analytical views. |
| **04 — Banking Intent Classifier** | A full fine-tune of ModernBERT-base on Banking77 with temperature scaling for calibration and a top-5 Gradio demo with confidence bars. |

---

# Project 1 — Research Assistant (RAG with LangGraph)

**Folder**: `01-LLM-RAG/research-assistant/`
**Live demo**: deploy with `bash scripts/deploy_hf_space.sh`
**Source**: full repo

## What it is
A Retrieval-Augmented Generation system. Given a research question, it ingests papers from ArXiv and the open web, indexes them in Chroma + BM25, and answers with a LangGraph workflow that rewrites the query, retrieves with hybrid search + cross-encoder reranking, grades the evidence, generates with inline citations, and self-reflects.

## Why it's interview-relevant
- **Modern LLM engineering pattern**: LangGraph state machines are the recommended way to build agentic flows in 2025–2026.
- **Production retrieval architecture**: hybrid dense + sparse + rerank is the industry standard for production RAG.
- **Engineering maturity**: multi-provider LLM factory (HuggingFace, Groq, Anthropic, OpenAI), Pydantic config, structured logging, RAGAS eval suite.

## Headline numbers / facts
- LangChain 0.3, LangGraph 0.2, ChromaDB, BM25, MiniLM cross-encoder reranker
- Default LLM: Llama-3.2-3B-Instruct via Hugging Face Inference (free)
- One reflection iteration cap; top-K is 5 after reranking

## What you actually built (in your own voice)
- The state machine: `rewrite_query → retrieve → grade → generate → reflect → (END or retry retrieve)`
- The provider factory in `src/llm/providers.py` — the only file that imports provider SDKs
- The hybrid retriever with dedup-by-content + cross-encoder reranking

## The hardest question they'll ask
*"Why LangGraph instead of an LCEL chain?"* — Because the flow isn't linear: depending on retrieval quality, we either generate or loop back. State machines are the right model for non-linear workflows, and LangGraph gives you streaming intermediate state for free.

---

# Project 2 — Dialogue Summarization (Llama-3.2 + LoRA)

**Folder**: `02-NLP-FineTuning/dialogue-summarization/`
**Artifacts**: LoRA adapter on HF Hub + Gradio Space
**Notebook**: `notebooks/01_finetune_samsum_lora.ipynb` (Colab-runnable)

## What it is
A QLoRA fine-tune of `meta-llama/Llama-3.2-3B-Instruct` on SAMSum (dialogue summarization, ~14.7k examples). The base is loaded in 4-bit NF4, LoRA adapters (r=16, α=32) are placed on q/k/v/o, and trained with TRL's SFTTrainer with `assistant_only_loss=True` so loss is only computed on summary tokens.

## Why it's interview-relevant
- **Modern fine-tuning recipe**: QLoRA is the standard for single-GPU LLM fine-tuning in 2025–2026.
- **Encoder-decoder → decoder-only migration**: the explicit "what changed" table in the README is great interview material.
- **Right tool for the regime**: paired with Project 4 to show you understand when LoRA vs full fine-tune is appropriate.

## Headline numbers / facts
- Trainable params: ~24M (~0.75% of total)
- Adapter size on disk: ~30 MB
- Quantization: 4-bit NF4 + double-quant + bf16 compute (QLoRA recipe)
- Trainer: TRL SFTTrainer, `assistant_only_loss=True`, `packing=False`
- LR 2e-4 cosine, warmup 0.03, batch 2 × grad-accum 4 = effective 8
- Time: ~25 min on a free Colab T4

## What you actually built
- The chat-template formatter that turns SAMSum rows into Llama-3 chat messages
- The QLoRA loader (`load_for_training`) with `prepare_model_for_kbit_training`
- The SFTTrainer config with `assistant_only_loss=True`
- The push-to-hub script that auto-generates the model card

## The hardest question they'll ask
*"Why `assistant_only_loss=True` matters."* — Without it, the loss is computed on every token in the sequence, including the system prompt and user message — wasted gradient signal. With it, only the model's response tokens contribute to loss, which is what you actually want to learn.

---

# Project 3 — LA Crime Interactive Dashboard

**Folder**: `03-EDA-Dashboards/la-crime-analysis/`
**Live demo**: Streamlit app, HF Spaces deployable
**Original artifact**: `notebook.ipynb` preserved in same folder

## What it is
An interactive Streamlit dashboard built on top of an earlier exploratory notebook. The notebook answered three fixed questions about LA crime; the dashboard turns the same data into a tool with interactive filters (date, area, crime type) that drive four analytical views (Overview, Geography, Time, Demographics).

## Why it's interview-relevant
- **Product thinking**: shows you understand that a notebook is not the same as a deliverable. Hiring managers care about this.
- **Caching and performance**: `@st.cache_data` is the kind of practical Streamlit knowledge that distinguishes you from someone who just ran `streamlit run` once.
- **Ethical framing**: the demographic-page warning about reported-victim vs predicted-offender data — interviewers notice this kind of thing.

## Headline numbers / facts
- 185k rows, 27 MB CSV, ~190k entries
- Streamlit 1.40 multi-page (via `pages/` convention)
- Plotly Express + Graph Objects for all interactive charts
- 4 pages, shared sidebar filters across all of them

## What you actually built
- `src/data.py:add_features` — the pure function that derives all the downstream columns. Unit-tested.
- `src/filters.py:sidebar_filters` — one widget set used across all pages so they stay consistent.
- The Plotly chart library in `src/viz.py` — every page composes from this.

## The hardest question they'll ask
*"What's the original notebook doing in the same folder?"* — Honest answer: it's evidence of the project's evolution. From "I learned pandas and answered three questions" to "I built a stakeholder-usable tool over the same data." For a portfolio, that progression *is* the story.

---

# Project 4 — Banking Intent Classifier (ModernBERT + Banking77)

**Folder**: `04-Deep-Learning/text-classification-modernbert/`
**Artifacts**: HF Hub model + HF Space demo

## What it is
A full fine-tune of `answerdotai/ModernBERT-base` (the 2024 BERT-family update) on `PolyAI/banking77` — 77-class customer-intent classification. Plus temperature scaling for calibration, so the demo's confidence bars actually mean something.

## Why it's interview-relevant
- **Discriminative transformer fine-tuning** — different signal from Project 2's generative LoRA work.
- **Calibration**: most fresh candidates don't know what temperature scaling is or why it matters. This sets you apart.
- **Honest evaluation**: stratified val split, test held out, fit T on val, report ECE before/after.

## Headline numbers / facts
- 149M params, encoder-only, 8k context, RoPE + GeGLU + alternating local/global attention
- Full fine-tune (not LoRA — deliberately contrasted with Project 2)
- Dataset: 13k train / 3k test, 77 classes; stratified 10% val split out of train
- Recipe: AdamW lr 5e-5 cosine, 10% warmup, weight-decay 0.01, label-smoothing 0.1, bf16, 4 epochs, batch 32
- Selection: macro F1 on val
- Calibration: temperature fit by NLL on val with 1-D ternary search (no torch dep)

## What you actually built
- `src/calibration.py` — three pure-numpy functions (fit T, apply T, ECE). Fully unit-tested.
- `src/eval.py` — the multi-metric evaluator that reports ECE before/after calibration.
- The Gradio app surfaces top-5 because semantically-near intents (e.g. `card_payment_fee_charged` vs `cash_withdrawal_charge`) are easy to confuse, and a real system would route to a human when the second probability is close to the first.

## The hardest question they'll ask
*"What is temperature scaling and why do you do it?"* — A modern over-parameterized classifier is overconfident: when it says "97% sure", it's right less often than that. T-scaling fits one scalar to rescale logits so confidence matches observed accuracy. Doesn't change predictions, only their meaning. Matters whenever something downstream thresholds on confidence.

---

# Cross-project questions

These are the questions interviewers actually ask. They're testing whether you understand the *space* of decisions, not just the projects.

## "Compare your two transformer fine-tuning projects."

> Project 2 is generative — decoder-only Llama-3.2-3B, instruction tuning, free-text outputs. There LoRA is the right tool: it cuts memory ~4× and trains under 1% of params with negligible quality loss. Project 4 is discriminative — encoder-only ModernBERT-base, 149M params, 77-class classification, fresh head. There full fine-tune is the right tool: the model is small enough that LoRA's memory savings don't matter, and fresh classification heads need to learn from scratch which LoRA can't help with. Picking the right method for the regime is half the job.

## "How do you decide between RAG and fine-tuning?"

> Different problems. RAG is for *up-to-date or proprietary knowledge* — when the answer depends on documents the base LLM has never seen. Fine-tuning is for *behavior* — when you want the model to consistently format outputs a certain way, follow a domain style, or specialize in a task. They compose: a production system often fine-tunes a model on the *style* of answers and uses RAG for the *content*. Project 1 is the RAG side; Project 2 is the fine-tuning side.

## "Walk me through how you'd build a customer-service chatbot."

> Four pieces. (a) An intent classifier like Project 4 routes the user's message to one of N intents. (b) For knowledge-based intents (FAQ, policy questions), RAG over the company's documentation — Project 1's architecture, scaled down. (c) For task-based intents (cancel a payment, change an address), structured tool-calling against internal APIs. (d) Underneath everything, observability: every turn logged with intent confidence and retrieval evidence, so you can debug failures and label data for future iterations.

## "What's your evaluation philosophy?"

> Three principles. **Hold out test data and don't touch it until the final report.** Project 4's stratified val/test split exists exactly for this. **Use multiple metrics that disagree about different things.** Project 2 reports both ROUGE (n-gram overlap, punishes paraphrase) and BERTScore (semantic, more forgiving) — when they disagree it's a signal. **Calibration matters if confidence matters.** Project 4 reports ECE before and after temperature scaling, and the demo uses calibrated confidence.

## "How would you deploy these to production (not portfolio HF Spaces)?"

> Three changes per project. (a) Move the inference layer behind a managed endpoint — HF Inference Endpoints, AWS SageMaker, or self-hosted on K8s with vLLM/TGI for LLMs, ONNX + Triton for the smaller models. (b) Add an observability layer: structured logs (already in Project 1), tracing (LangSmith for the LLM path, OpenTelemetry for the API path), and request IDs. (c) Add real auth and rate limiting — none of the demos have either, because they're public portfolio pieces, but you obviously can't ship like that. For the dashboard specifically, I'd swap the in-repo CSV for Parquet on S3 with pre-aggregations in DuckDB / Postgres so it scales past the in-memory limit.

## "What would you build next?"

> Three honest answers. (a) An MLOps project tying everything together — MLflow tracking, DVC for data versioning, GitHub Actions for CI/CD with training in the pipeline, monitoring with Prometheus + Grafana. (b) A multi-modal RAG that handles PDFs with figures, not just text — academic papers lose 30%+ of their information when you strip to text. (c) An online evaluation system: instead of offline RAGAS, instrument the actual demo and use the data to A/B test prompt and retrieval changes.

## "What's the project you're least happy with?"

> Honest answer: probably the LA Crime dashboard. It's the most polished as a *product* but the least technically deep. I kept it in the portfolio because it shows a different signal — turning ambiguous data into something a stakeholder can actually use — but if I had another week I'd add a forecasting model or area-clustering on top to give it more ML signal beyond viz.

## "What's the project you're proudest of?"

> The research assistant (Project 1). LangGraph state machines are genuinely modern and the pattern matters. The multi-provider LLM factory is the kind of design I'd be proud to put in front of a senior engineer. And the INTERVIEW_NOTES.md is the most worked-through of the four — that's the one I can defend deepest.

## "How long did this take you?"

> Be honest about this — interviewers have a good sense for whether portfolio work was over-claimed. The truthful answer is something like "I built and documented this monorepo over [N] days with Claude Code as a coding pair. The architectural decisions, the choice of stack, and the trade-offs I can defend in detail are mine; the boilerplate scaffolding and a chunk of the typing-out is Claude's. Knowing how to use AI tooling efficiently is itself part of the modern ML engineering job."

---

# Quick reference — facts to know cold

## Versions & libraries (current as of project build)
- Python 3.10+ (3.11 in Docker images)
- LangChain 0.3, LangGraph 0.2, LangChain-Chroma 0.1
- Transformers 4.46+ (4.48 for ModernBERT), PEFT 0.13, TRL 0.12, Accelerate 1.1, bitsandbytes 0.44
- Streamlit 1.40, Plotly 5.24
- Gradio 5
- FastAPI 0.115, Pydantic 2.9
- scikit-learn 1.5

## Models
- LLM (default): `meta-llama/Llama-3.2-3B-Instruct`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Classifier base: `answerdotai/ModernBERT-base` (149M, 8k ctx)

## Datasets
- `samsum` (~14.7k train / 819 val / 818 test) — Project 2
- `PolyAI/banking77` (13k / 3k, 77 classes) — Project 4
- LA Crime (LAPD via LA Open Data, 185k rows) — Project 3

## Eval metrics
- **Generation**: ROUGE-1/2/L (rouge-score) + BERTScore-F1
- **RAG**: RAGAS (faithfulness, answer-relevancy, context-precision, context-recall)
- **Classification**: accuracy, macro F1, top-K accuracy, ECE (15-bin)

## Deployment targets
- **Hugging Face Spaces** (Docker SDK) for all four — one bash script per project
- All projects have a Dockerfile, a `.env.example`, and a `scripts/deploy_hf_space.sh`

---

# What to do the night before an interview

1. **Re-read the project's own `docs/INTERVIEW_NOTES.md`** — it's denser than this summary.
2. **Open `docs/ARCHITECTURE.md`** for the project they'll likely ask about — be able to draw the diagram from memory.
3. **Practice the one-sentence pitch out loud.** If the words don't come naturally, you need another pass.
4. **Open one specific code file** for each project and rehearse the "show me one specific design decision" answer:
   - Project 1: `src/graph/workflow.py:build_graph` (the conditional edge in `reflect`)
   - Project 2: `src/data.py:_format_row` (chat-template alignment)
   - Project 3: `src/data.py:add_features` (pure function -> testable, swappable storage)
   - Project 4: `src/calibration.py:fit_temperature` (one scalar that turns confidence numbers from decorative into honest)
5. **Have the HF Space URLs in your tab bar** so you can demo on request.

---

# What NOT to do in an interview

- **Don't claim the AI wrote nothing.** Modern interviewers ask about your AI workflow specifically. Saying "I built every line myself" when you didn't is detectable and damaging. Saying "I used Claude Code as a pair, here's what I directed vs. what was generated" is honest and increasingly valued.
- **Don't memorize every line of code.** You can't, and you don't need to. Memorize *decisions* — the trade-offs you considered and the reasons you chose what you chose. Decisions are what interviewers actually probe.
- **Don't oversell.** "Production-grade" only if you can defend it. "Inspired by production patterns" or "production-shaped" is honest and still impressive.
- **Don't dodge "what's wrong with this project."** Every project has limitations. Calling them out shows engineering maturity. The README of each project already lists "what's missing and why" — quote yourself.
