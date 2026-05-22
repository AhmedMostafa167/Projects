"""Push a trained ModernBERT classifier (model + tokenizer + model card) to the HF Hub.

Run:
    python scripts/push_to_hub.py \\
        --model_dir ./outputs/modernbert-banking77 \\
        --repo_id <username>/modernbert-banking77
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
library_name: transformers
base_model: {base_model_id}
tags:
  - text-classification
  - intent-classification
  - banking
  - modernbert
datasets:
  - PolyAI/banking77
language:
  - en
pipeline_tag: text-classification
---

# ModernBERT fine-tuned on Banking77

Fine-tune of `{base_model_id}` for 77-class customer-intent classification.

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="{repo_id}", top_k=5)
clf("My card hasn't arrived yet, what should I do?")
```

## Training

- **Base model**: `{base_model_id}` (149M params, encoder-only, 8k context)
- **Dataset**: PolyAI/banking77 (~13k train / 3k test, 77 classes)
- **Recipe**: full fine-tune, lr=5e-5 cosine schedule with 10% warmup, 4 epochs,
  bf16, weight-decay=0.01, label-smoothing=0.1, batch size 32
- **Validation split**: 10% of train, stratified by label
- **Metric for best model**: macro F1

## Calibration

Logits are saved as-is. To get calibrated probabilities, divide logits by the
fitted temperature (reported in `eval_results.json` in the source repo) before
softmax. The fitted T value is typically slightly > 1, meaning the raw model is
mildly overconfident.

## Source

[GitHub: AhmedMostafa167/Projects/04-Deep-Learning/text-classification-modernbert](https://github.com/AhmedMostafa167/Projects/tree/main/04-Deep-Learning/text-classification-modernbert)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"Model dir not found: {args.model_dir}")

    from huggingface_hub import HfApi

    from src.config import settings

    api = HfApi(token=settings.huggingfacehub_api_token)

    print(f"Creating repo {args.repo_id}...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    card_path = args.model_dir / "README.md"
    card_path.write_text(
        MODEL_CARD_TEMPLATE.format(base_model_id=settings.base_model_id, repo_id=args.repo_id)
    )
    print(f"Wrote model card -> {card_path}")

    print("Uploading model folder...")
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.model_dir),
        commit_message="Initial model upload",
    )
    print(f"Done. https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
