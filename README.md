# ML / NLP Portfolio — Ahmed Mostafa

A curated set of machine-learning and natural-language-processing projects, each self-contained, Dockerized, deployable, and documented for technical discussion.

> **Status**: actively building. This README is the index — click into any project for its full README, architecture notes, and interview talking points.

---

## Project index

| # | Project | Domain | Stack | Live demo | Source |
|---|---|---|---|---|---|
| 01 | **Research Assistant** | LLM, RAG, agentic workflows | LangChain 0.3, LangGraph, Chroma, FastAPI, Gradio | _(deploy ready)_ | [`01-LLM-RAG/research-assistant/`](01-LLM-RAG/research-assistant/) |
| 02 | **Dialogue Summarization** | NLP, transformer fine-tuning, PEFT | Transformers 4.46, PEFT, TRL, bitsandbytes, HF Hub, Gradio | _(train + deploy ready)_ | [`02-NLP-FineTuning/dialogue-summarization/`](02-NLP-FineTuning/dialogue-summarization/) |
| 03 | **LA Crime Interactive Dashboard** | EDA, product thinking, deployment | Streamlit 1.40, Plotly, pandas | _(deploy ready)_ | [`03-EDA-Dashboards/la-crime-analysis/`](03-EDA-Dashboards/la-crime-analysis/) |
| 04 | **Banking Intent Classifier** | NLP, discriminative transformer fine-tuning, calibration | ModernBERT, PyTorch, scikit-learn | _(train + deploy ready)_ | [`04-Deep-Learning/text-classification-modernbert/`](04-Deep-Learning/text-classification-modernbert/) |

---

## What this portfolio is meant to show

For an NLP / ML engineering role, recruiters and hiring managers usually look for evidence across four axes. Each project is positioned against one of these axes:

| Axis | Project that shows it |
|---|---|
| **Modern LLM / agentic engineering** | 01 — Research Assistant (LangGraph, hybrid retrieval, multi-provider LLM design, evaluation) |
| **Foundational NLP / transformer fine-tuning** | 02 — Dialogue Summarization (PEFT/LoRA, HF Transformers, model card on the Hub) |
| **Data exploration & communication** | 03 — LA Crime Dashboard (interactive viz, deployed) |
| **Classical deep learning** | 04 — PyTorch project |

---

## Repo conventions

Every project under this repo follows the same structure so they're easy to read in any order:

```
NN-Category/<project-name>/
├── README.md            # What it is, why it exists, how to run, what to look at
├── requirements.txt     # Pinned dependencies
├── pyproject.toml       # Modern Python packaging
├── Dockerfile           # Self-contained container
├── docker-compose.yml   # Local orchestration if multi-service
├── .env.example         # All config knobs documented
├── src/                 # Library code
├── tests/               # Pytest suite
├── notebooks/           # Exploration notebooks
├── scripts/             # Deploy / utility scripts
└── docs/
    ├── ARCHITECTURE.md  # System design
    ├── DEPLOYMENT.md    # How to deploy
    └── INTERVIEW_NOTES.md  # What to talk about in interviews
```

The `docs/INTERVIEW_NOTES.md` files are intentionally written as a cheat-sheet I can study before interviews — likely questions, 30-second answers, and deeper context.

---

## Running anything locally

Each project is self-contained:

```bash
cd <project-folder>
pip install -e ".[dev]"
cp .env.example .env  # fill in any required keys
python app.py         # or whatever the project's entry point is
```

Or with Docker:

```bash
cd <project-folder>
docker compose up --build
```

---

## Original / archived repos

These were earlier learning repos kept on the same GitHub account; the modernized versions live in this monorepo:

- [`langchain-research-assistant`](https://github.com/AhmedMostafa167/langchain-research-assistant) → modernized as **01-LLM-RAG/research-assistant**
- [`PEGASUS-DialogueSum`](https://github.com/AhmedMostafa167/PEGASUS-DialogueSum) → modernized as **02-NLP-FineTuning** (coming)
- [`Pytorch-Exercises-and-Projects`](https://github.com/AhmedMostafa167/Pytorch-Exercises-and-Projects) → cherry-picked into **04-Deep-Learning** (coming)
