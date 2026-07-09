# Deployment

## Step 1 — Fine-tune (Colab / Kaggle, free GPU)

```bash
pip install -e ".[train]"
huggingface-cli login   # only needed if you want to push the result
python -m src.train --output_dir ./outputs/modernbert-banking77
```

Or open `notebooks/02_finetune_banking77.ipynb` on Colab. Time: ~15 min on T4.

---

## Step 2 — Evaluate + fit temperature

```bash
python -m src.eval \
    --model_dir ./outputs/modernbert-banking77 \
    --output ./outputs/eval_results.json
```

Writes the full metrics (acc, macro F1, top-3, top-5, ECE before/after T) plus the per-class classification report. Update the README results table with these numbers.

---

## Step 3 — Push to HF Hub

```bash
python scripts/push_to_hub.py \
    --model_dir ./outputs/modernbert-banking77 \
    --repo_id <your-username>/modernbert-banking77
```

Uploads the model, tokenizer, and a generated model card.

---

## Step 4 — Deploy the demo

```bash
cp .env.example .env
# Edit: set HF_MODEL_REPO=<your-username>/modernbert-banking77
bash scripts/deploy_hf_space.sh <your-username> modernbert-banking77-demo
```

Creates a Docker-SDK HF Space, uploads the code, uploads `.env` values as secrets. Live in ~3 minutes.

---

## Local options

### Python
```bash
pip install -e .
cp .env.example .env
HF_MODEL_REPO=<your-username>/modernbert-banking77 python app.py
# → http://localhost:7860
```

If you haven't trained yet, leaving `HF_MODEL_REPO` blank loads base ModernBERT with a randomly-initialized head — the demo runs but predictions are nonsense. Useful for smoke-testing the Gradio plumbing.

### Docker
```bash
docker build -t modernbert-banking77 .
docker run -p 7860:7860 --env-file .env modernbert-banking77
```

---

## Hardware

- **Training**: needs CUDA GPU with ≥8 GB VRAM. T4 is fine; A10/A100 are faster but unnecessary.
- **Inference**: CPU works (~150 ms / query on a modern laptop), GPU is faster but optional.

---

## Cost

| Step | Cost |
|---|---|
| Training (4 epochs) on Colab T4 | $0 |
| Model on HF Hub | $0 |
| Demo on HF Space (CPU) | $0 |

Total: $0 for a deployed fine-tuned classifier.
