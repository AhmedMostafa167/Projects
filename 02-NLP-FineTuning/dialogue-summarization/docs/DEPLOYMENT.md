# Deployment

This project ships in two pieces: a **model artifact** (LoRA adapter on the HF Hub) and a **demo** (Gradio Space). Below is the full path from a trained adapter to a public demo.

---

## Step 1 — Fine-tune (Colab / Kaggle, free)

Open `notebooks/01_finetune_samsum_lora.ipynb` on Colab. Make sure the runtime is set to GPU (T4 is fine). The notebook handles login, training, and saving the adapter to `./outputs/llama32-samsum-lora/`.

**Time**: ~25 min for 2 epochs.

If you prefer the CLI:
```bash
pip install -e ".[train]"
huggingface-cli login
python -m src.train --output_dir ./outputs/llama32-samsum-lora
```

---

## Step 2 — Push the adapter to the HF Hub

```bash
python scripts/push_to_hub.py \
    --adapter_dir ./outputs/llama32-samsum-lora \
    --repo_id <your-username>/llama32-samsum-lora
```

This:
1. Creates a model repo on the Hub.
2. Generates a model card with the training recipe.
3. Uploads the adapter (~30 MB).

The repo URL will be `https://huggingface.co/<your-username>/llama32-samsum-lora`.

---

## Step 3 — Deploy the Gradio demo to HF Spaces

```bash
cd 02-NLP-FineTuning/dialogue-summarization
cp .env.example .env
# Edit .env: set HF_ADAPTER_REPO=<your-username>/llama32-samsum-lora
bash scripts/deploy_hf_space.sh <your-hf-username> dialogue-summarization
```

The script creates a Docker-SDK Space, swaps in the Space-formatted README header, uploads the project, and pushes each `.env` value as a Space Secret.

Your Space will be live at `https://huggingface.co/spaces/<your-username>/dialogue-summarization` within ~5 minutes.

---

## Local options

### Plain Python

```bash
pip install -e .
cp .env.example .env  # set HF_ADAPTER_REPO
python app.py
# → http://localhost:7860
```

If you don't have a fine-tuned adapter yet, leave `HF_ADAPTER_REPO` blank — the app falls back to the zero-shot base model so the demo is always runnable.

### Docker

```bash
docker build -t dialogue-summarization .
docker run -p 7860:7860 --env-file .env dialogue-summarization
```

---

## Hardware notes

- **Training**: needs a CUDA GPU with ≥16 GB VRAM (T4, V100, A10, A100, RTX 3090+). With QLoRA on a 3B model, peak memory is ~10–12 GB.
- **Inference (demo)**: CPU works but is slow (~30 s per summary). For a snappier demo, upgrade the Space to a small GPU ($0.40/hr at the time of writing) or use the HF Inference API instead of loading the model in-process.

---

## Cost summary

| Step | Where | Cost |
|---|---|---|
| Training (2 epochs) | Colab free T4 | $0 |
| Adapter on HF Hub | HF Hub | $0 (public repos free) |
| Demo on HF Space (CPU) | HF Spaces free tier | $0 |
| Demo on HF Space (T4) | HF Spaces hardware upgrade | ~$0.40/hr only when used |

**Total to get a fully deployed portfolio piece: $0.**
