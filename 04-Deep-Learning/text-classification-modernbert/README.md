# Banking Intent Classification — ModernBERT on Banking77

> Fine-tune ModernBERT (the 2024 successor to BERT/DeBERTa, with 8k context and modern training tricks) on Banking77 — a 77-way customer-intent classification benchmark — and ship a calibrated Gradio demo with top-K predictions and confidence scores.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Transformers](https://img.shields.io/badge/Transformers-4.46-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)

---

## Why this project (and how it's different from Project 2)

| | Project 2: Dialogue Summarization | This project: Banking77 Intent Classification |
|---|---|---|
| Task type | **Generative** (free-text output) | **Discriminative** (pick 1 of 77 classes) |
| Architecture | Decoder-only (Llama-3.2-3B) | Encoder-only (ModernBERT-base) |
| Fine-tuning method | **LoRA** (parameter-efficient) | **Full fine-tune** of all 149M params |
| Trainer | `trl.SFTTrainer` | `transformers.Trainer` |
| Eval metrics | ROUGE, BERTScore | Macro F1, top-K accuracy, calibration (ECE) |
| Why both | Real ML engineering means knowing when each tool applies |

A fresh NLP/ML engineer is expected to be fluent in **both** patterns. Pairing these two projects shows you understand the difference.

---

## Why ModernBERT?

`answerdotai/ModernBERT-base` is the 2024 modernization of the BERT encoder family from Answer.AI. Compared to BERT-base:
- **8192-token context** (vs. 512) — handles long inputs without sliding windows.
- **Rotary position embeddings (RoPE)** — better length extrapolation than learned positional embeddings.
- **GeGLU activations**, **alternating local/global attention** — significantly improved NLU benchmark scores.
- **Same 149M params** as DeBERTa-v3-base, but trained on much more data (~2T tokens).

In practical terms: it's a drop-in replacement for any place you'd use BERT/RoBERTa/DeBERTa, with better accuracy and much longer context.

For Banking77, where inputs are short customer queries, the long-context advantage doesn't kick in — but the improved representation quality still helps. The expected macro F1 is in the **92–94%** range on the standard test split.

---

## Why Banking77?

It's the standard fine-grained intent classification benchmark:
- ~13k train / 3k test examples.
- 77 fine-grained classes ("card_arrival", "card_payment_fee_charged", "card_payment_not_recognised"...).
- Realistic — these are the intents a real banking chatbot would route on.
- Hard enough to be interesting: many classes are semantically very close.

---

## Quickstart

### Fine-tune (Colab GPU, free)

```bash
git clone https://github.com/AhmedMostafa167/Projects.git
cd Projects/04-Deep-Learning/text-classification-modernbert
pip install -e ".[train]"
python -m src.train --output_dir ./outputs/modernbert-banking77
```

Or open `notebooks/01_finetune_banking77.ipynb` on Colab — same script, with explanatory cells.

### Push to HF Hub

```bash
python scripts/push_to_hub.py \
    --model_dir ./outputs/modernbert-banking77 \
    --repo_id <your-username>/modernbert-banking77
```

### Run the demo

```bash
HF_MODEL_REPO=<your-username>/modernbert-banking77 python app.py
# → http://localhost:7860
```

### Deploy to HF Spaces

```bash
bash scripts/deploy_hf_space.sh <your-hf-username> modernbert-banking77
```

---

## Project layout

```
text-classification-modernbert/
├── app.py                    # Gradio demo: top-K predictions + confidence
├── src/
│   ├── config.py             # Pydantic settings
│   ├── data.py               # Banking77 loader + tokenization
│   ├── model.py              # AutoModelForSequenceClassification wrapper
│   ├── train.py              # Trainer-based training entry point
│   ├── eval.py               # Macro F1, top-K accuracy, ECE, confusion matrix
│   ├── calibration.py        # Temperature scaling for honest confidence scores
│   └── infer.py              # Inference + top-K helper
├── notebooks/
│   ├── 01_baselines.ipynb           # Zero-shot + TF-IDF + LogReg baselines
│   ├── 02_finetune_banking77.ipynb  # Main fine-tuning walkthrough
│   └── 03_error_analysis.ipynb      # Confusion matrix + per-class breakdown
├── scripts/
│   ├── push_to_hub.py
│   └── deploy_hf_space.sh
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── INTERVIEW_NOTES.md
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Tech stack

- **Base model**: `answerdotai/ModernBERT-base` (149M params, 8k context)
- **Dataset**: `PolyAI/banking77` via HF Datasets
- **Training**: `transformers.Trainer`, mixed precision (bf16), cosine LR schedule with warmup, weight decay
- **Evaluation**: scikit-learn metrics (macro F1, accuracy, top-K) + custom ECE
- **Calibration**: temperature scaling on a held-out split
- **Inference**: HF Pipelines (`text-classification` with `top_k`)
- **UI**: Gradio 5 with a confidence bar chart
- **Distribution**: HF Hub (model + tokenizer) + HF Spaces (demo)

---

## Results

(Run the training notebook; the eval script writes the table below into `outputs/eval_results.json`.)

| Model | Accuracy | Macro F1 | Top-3 Accuracy | ECE |
|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Zero-shot (NLI-based) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ModernBERT (this work) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ModernBERT + temp scaling | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

See [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) for talking points.
