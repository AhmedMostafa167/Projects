# Deployment

## Option 1 — Hugging Face Spaces (recommended, free)

**One-time setup:**

```bash
pip install -U huggingface_hub
huggingface-cli login  # paste a write token from https://huggingface.co/settings/tokens
```

**Deploy:**

```bash
bash scripts/deploy_hf_space.sh <your-hf-username> research-assistant
```

The script creates the Space (Docker SDK), uploads the code, swaps in the Space-formatted README header, and uploads each key in `.env` as a Space Secret. Your Space will be live at `https://huggingface.co/spaces/<your-hf-username>/research-assistant` within a few minutes (first build is ~5–10 min).

**Resource notes:**
- HF Spaces free tier: 2 vCPU, 16 GB RAM, CPU only.
- The `all-MiniLM-L6-v2` embedding model and `ms-marco-MiniLM-L-6-v2` reranker both fit comfortably.
- Don't try to load a 7B+ LLM locally — use the HF Inference API or Groq.

---

## Option 2 — Render / Railway (FastAPI backend)

Both platforms accept the `Dockerfile` as-is. For Render:
1. New Web Service → connect this repo, set root directory to `01-LLM-RAG/research-assistant`.
2. Set the start command to: `uvicorn api:app --host 0.0.0.0 --port $PORT`
3. Add each env var from `.env.example` in the Render dashboard.

For a Gradio-only deployment, leave the Dockerfile's default `CMD ["python", "app.py"]` and set `PORT=7860`.

---

## Option 3 — Docker locally

```bash
docker compose up --build
# Gradio:  http://localhost:7860
# FastAPI: http://localhost:8000/docs
```

---

## Option 4 — Local Python

```bash
pip install -e ".[dev]"
cp .env.example .env  # then fill in HUGGINGFACEHUB_API_TOKEN
python app.py
```

---

## Switching LLM providers

Edit `.env`:

```bash
LLM_PROVIDER=groq   # or anthropic, openai, huggingface
GROQ_API_KEY=gsk_...
```

Then install the provider extra:

```bash
pip install -e ".[groq]"     # or [anthropic] / [openai]
```

That's the only change — the rest of the code is provider-agnostic.

---

## Troubleshooting

**The Space build fails with "out of memory":** the embedding model loads on import; if you've enabled the reranker on a very small Space, it can OOM. Solution: upgrade to a paid Space, or disable the reranker by setting `TOP_K_RERANK` ≥ the sum of dense + sparse top-K so the reranker becomes a no-op.

**HuggingFace Inference returns 429:** rate-limited. Either wait, or switch `LLM_PROVIDER` to Groq (separate free tier).

**Cold start is slow:** first request downloads ~200MB of models. The Docker image pre-warms `HF_HOME` to `/app/.hf_cache`, and the compose file mounts that as a named volume so subsequent restarts are fast.
