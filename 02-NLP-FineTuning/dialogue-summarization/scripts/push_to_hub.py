"""Push a trained LoRA adapter (plus a generated model card) to the HF Hub.

Run:
    python scripts/push_to_hub.py \\
        --adapter_dir ./outputs/llama32-samsum-lora \\
        --repo_id <username>/llama32-samsum-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
library_name: peft
base_model: {base_model_id}
tags:
  - dialogue-summarization
  - lora
  - peft
  - samsum
datasets:
  - samsum
metrics:
  - rouge
language:
  - en
---

# Llama-3.2-3B-Instruct (LoRA) on SAMSum

LoRA adapter for `{base_model_id}` fine-tuned on the [SAMSum](https://huggingface.co/datasets/samsum) dialogue summarization dataset.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model_id}")
tok = AutoTokenizer.from_pretrained("{base_model_id}")
model = PeftModel.from_pretrained(base, "{repo_id}")

messages = [
    {{"role": "system", "content": "You are a concise dialogue-summarization assistant. Given a conversation, produce a short, neutral summary in 1-3 sentences."}},
    {{"role": "user", "content": "Summarize this conversation:\\n\\n<your dialogue here>"}},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

- **Base model**: `{base_model_id}` (loaded in 4-bit NF4)
- **Method**: QLoRA (rank 16, alpha 32, dropout 0.05)
- **Target modules**: q_proj, k_proj, v_proj, o_proj
- **Trainer**: `trl.SFTTrainer` with `assistant_only_loss=True`
- **Optimizer**: AdamW, lr 2e-4, cosine schedule, warmup ratio 0.03
- **Epochs**: 2
- **Effective batch size**: 8 (per-device 2 × grad-accum 4)
- **Max sequence length**: 1024

## Demo

[Live Gradio Space](https://huggingface.co/spaces) — see the project README for the URL.

## Source

[GitHub: AhmedMostafa167/Projects/02-NLP-FineTuning/dialogue-summarization](https://github.com/AhmedMostafa167/Projects/tree/main/02-NLP-FineTuning/dialogue-summarization)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter dir not found: {args.adapter_dir}")

    from huggingface_hub import HfApi

    from src.config import settings

    api = HfApi(token=settings.huggingfacehub_api_token)

    print(f"Creating repo {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    card_path = args.adapter_dir / "README.md"
    card_path.write_text(
        MODEL_CARD_TEMPLATE.format(base_model_id=settings.base_model_id, repo_id=args.repo_id)
    )
    print(f"Wrote model card -> {card_path}")

    print("Uploading adapter folder...")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.adapter_dir),
        commit_message="Initial adapter upload",
    )
    print(f"Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
