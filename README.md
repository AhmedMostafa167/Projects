# ML / NLP Portfolio — Ahmed Mostafa

A curated set of machine-learning and natural-language-processing projects, each self-contained, Dockerized, deployable, and documented for technical discussion.

> **Status**: 4 projects in this monorepo, each with a structured layout and a one-command deploy script for Hugging Face Spaces.

---

## Project index

| # | Project | Domain | Stack | Status | Source |
|---|---|---|---|---|---|
| 01 | **Research Assistant** | LLM, RAG, agentic workflows | LangChain 0.3, LangGraph, Chroma, FastAPI, Gradio | Deploy-ready | [`01-LLM-RAG/research-assistant/`](01-LLM-RAG/research-assistant/) |
| 02 | **Dialogue Summarization** | NLP, generative fine-tuning, PEFT | Llama-3.2-3B, QLoRA, TRL, HF Hub, Gradio | Train + deploy ready | [`02-NLP-FineTuning/dialogue-summarization/`](02-NLP-FineTuning/dialogue-summarization/) |
| 03 | **LA Crime Dashboard** | EDA, product thinking, interactive viz | Streamlit 1.40, Plotly, pandas | Deploy-ready | [`03-EDA-Dashboards/la-crime-analysis/`](03-EDA-Dashboards/la-crime-analysis/) |
| 04 | **Banking Intent Classifier** | NLP, discriminative fine-tuning, calibration | ModernBERT, PyTorch, scikit-learn | Train + deploy ready | [`04-Deep-Learning/text-classification-modernbert/`](04-Deep-Learning/text-classification-modernbert/) |

---

## Start here

Pick any project above — each has its own README with a quickstart, architecture diagram, and a link to the live demo. Every project folder also contains:

- `docs/ARCHITECTURE.md` — the system design
- `docs/DEPLOYMENT.md` — exact commands for running and deploying

---

## What this portfolio shows

| Axis | Project | Signal |
|---|---|---|
| **Modern LLM / agentic engineering** | 01 — Research Assistant | LangGraph state machines, hybrid retrieval, multi-provider LLM abstraction, RAGAS eval |
| **Generative transformer fine-tuning** | 02 — Dialogue Summarization | QLoRA (4-bit + LoRA r=16), TRL SFTTrainer with assistant-only loss, chat-template alignment |
| **Data exploration → product** | 03 — LA Crime Dashboard | Notebook → multi-page Streamlit app with shared filters, ethical framing |
| **Discriminative transformer fine-tuning** | 04 — Banking Intent Classifier | Full fine-tune of ModernBERT-base, temperature scaling for honest confidence, multi-metric eval |

Projects 02 and 04 are deliberately paired to show **method-selection awareness**: LoRA for generative tasks at 3B scale, full fine-tune for discriminative tasks at 149M scale.

---

## Repository conventions

Every project under this repo follows the same structure:

```
NN-Category/<project-name>/
├── README.md            # What it is, why it exists, how to run
├── requirements.txt     # Pinned dependencies
├── pyproject.toml       # PEP 621 packaging
├── Dockerfile           # Self-contained container
├── .env.example         # All config knobs documented
├── .gitignore
├── app.py               # Main entry point (Streamlit / Gradio / FastAPI)
├── src/                 # Library code
├── tests/               # Pytest suite
├── notebooks/           # Walkthrough notebooks (if any)
├── scripts/
│   ├── deploy_hf_space.sh     # One-command HF Spaces deploy
│   └── push_to_hub.py         # (if the project produces a model)
└── docs/
    ├── ARCHITECTURE.md
    └── DEPLOYMENT.md
```

---

## Running anything locally

```bash
cd <project-folder>
pip install -e ".[dev]"
cp .env.example .env  # fill in any required keys (mostly HF tokens)
python app.py         # entry point varies — see project README
```

Or with Docker:

```bash
cd <project-folder>
docker build -t <project-name> .
docker run -p 7860:7860 --env-file .env <project-name>
```

---

## Deploying everything

Each project ships a single bash script:

```bash
# After: pip install -U huggingface_hub && huggingface-cli login
cd <project-folder>
bash scripts/deploy_hf_space.sh <your-hf-username> <space-name>
```

For Projects 02 and 04 (which produce a trained model), there's also `scripts/push_to_hub.py` for pushing the model to the HF Hub before deploying the demo.

See each project's `docs/DEPLOYMENT.md` for full details and alternative targets (Render, Railway, Streamlit Cloud).

---

## Original / archived repos

These were earlier learning repos kept on the same GitHub account; the modernized versions live in this monorepo.

- [`langchain-research-assistant`](https://github.com/AhmedMostafa167/langchain-research-assistant) → modernized as **01-LLM-RAG/research-assistant**
- [`PEGASUS-DialogueSum`](https://github.com/AhmedMostafa167/PEGASUS-DialogueSum) → reworked as **02-NLP-FineTuning/dialogue-summarization** (decoder-only + LoRA)
- [`Pytorch-Exercises-and-Projects`](https://github.com/AhmedMostafa167/Pytorch-Exercises-and-Projects) → derived into **04-Deep-Learning/text-classification-modernbert**

---

## License

MIT (per project — see each project's pyproject.toml).
