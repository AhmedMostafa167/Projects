# Dialogue Summarization — Llama-3.2-3B + LoRA on SAMSum

> A revisit of my earlier PEGASUS-based dialogue summarization project, rebuilt with modern parameter-efficient fine-tuning. Fine-tunes Llama-3.2-3B-Instruct on SAMSum using LoRA, evaluates with ROUGE + BERTScore, pushes the adapter to the Hugging Face Hub, and ships a Gradio demo.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Transformers](https://img.shields.io/badge/Transformers-4.46-yellow)
![PEFT](https://img.shields.io/badge/PEFT-0.13-orange)
![TRL](https://img.shields.io/badge/TRL-0.12-red)

---

## What's new vs. my earlier PEGASUS work

| Dimension | Old PEGASUS project | This project |
|---|---|---|
| Base model | PEGASUS-large (568M, encoder-decoder) | Llama-3.2-3B-Instruct (decoder-only) |
| Fine-tuning | Full fine-tune | LoRA (PEFT) — ~0.5% of params trained |
| Trainer | `transformers.Trainer` | `trl.SFTTrainer` (the modern instruction-tuning workflow) |
| Evaluation | ROUGE only | ROUGE-1/2/L + BERTScore |
| Artifact | Local checkpoint | Adapter pushed to HF Hub + Gradio Space |
| Hardware | Required full GPU memory for fine-tune | Runs on a free Colab T4 with 4-bit base + LoRA |

---

## Why this approach

**LoRA over full fine-tuning** — In 2024–2026 the field moved decisively to parameter-efficient fine-tuning. LoRA injects low-rank trainable matrices (`A`, `B`) into the attention layers, freezes the base model, and trains ~0.5% of total parameters. The base model can be kept in 4-bit precision (QLoRA), which cuts memory roughly 4× and makes single-GPU fine-tuning of 3B–7B models trivial. The fine-tuned adapter is ~30 MB instead of a multi-GB checkpoint.

**Decoder-only over encoder-decoder** — PEGASUS was state of the art in 2020 for abstractive summarization. In 2026 the strong baseline is instruction-tuned decoder-only LLMs, because (a) they generalize better to new prompt formats, (b) they share the same architecture and tokenizer as the LLMs used for everything else in a production stack, (c) modern alignment data is overwhelmingly instruction-format.

**SFTTrainer over Trainer** — `trl.SFTTrainer` is the canonical workflow for supervised instruction fine-tuning. It handles chat-template formatting, response-only loss masking, and plays nicely with PEFT.

---

## Quickstart

### Fine-tune (Colab / Kaggle free GPU)

```bash
git clone https://github.com/AhmedMostafa167/Projects.git
cd Projects/02-NLP-FineTuning/dialogue-summarization
pip install -e ".[train]"
huggingface-cli login   # need this to load Llama-3.2 (gated) + push the adapter
python -m src.train --output_dir ./outputs/llama32-samsum-lora
```

Or open `notebooks/01_finetune_samsum_lora.ipynb` and run on a Colab T4 — the notebook is the same script with explanations between cells.

### Push the adapter to the Hub

```bash
python scripts/push_to_hub.py \
  --adapter_dir ./outputs/llama32-samsum-lora \
  --repo_id <your-hf-username>/llama32-samsum-lora
```

### Run the Gradio demo

```bash
HF_ADAPTER_REPO=<your-hf-username>/llama32-samsum-lora python app.py
# → http://localhost:7860
```

### Deploy the demo to HF Spaces

```bash
bash scripts/deploy_hf_space.sh <your-hf-username> dialogue-summarization
```

---

## Project layout

```
dialogue-summarization/
├── app.py                    # Gradio demo (loads adapter from HF Hub)
├── src/
│   ├── config.py             # Pydantic settings: model id, LoRA params, etc.
│   ├── data.py               # SAMSum loading, chat-template formatting
│   ├── model.py              # Base model + LoRA adapter loader (with 4-bit option)
│   ├── train.py              # SFTTrainer-based training entry point
│   ├── infer.py              # Generation utilities
│   └── eval.py               # ROUGE + BERTScore evaluation
├── notebooks/
│   ├── 01_finetune_samsum_lora.ipynb    # Walkthrough (Colab-friendly)
│   └── 02_evaluation_comparison.ipynb   # Base vs fine-tuned, side-by-side
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

- **Base model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-tuning**: PEFT 0.13 (LoRA, r=16, α=32), TRL 0.12 (SFTTrainer)
- **Quantization**: bitsandbytes 4-bit NF4 (QLoRA)
- **Dataset**: `samsum` (~14.7k train / 819 val / 818 test)
- **Eval**: ROUGE-1/2/L (rouge-score), BERTScore
- **Inference UI**: Gradio 5
- **Distribution**: HF Hub (adapter weights) + HF Spaces (demo)

---

## Results (placeholder until training completes)

Once you run the fine-tune, the metrics get written here automatically by `src/eval.py`:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct (zero-shot) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Llama-3.2-3B-Instruct + LoRA (this work) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| PEGASUS-large (original 2020 baseline) | 0.42 | 0.18 | 0.34 | — |

The expected outcome: LoRA-tuned Llama-3.2-3B should match or beat PEGASUS-large on ROUGE while needing ~4× less GPU memory at inference (4-bit) and producing more coherent, instruction-following summaries.

---

See [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) for talking points.
