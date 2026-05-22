"""Evaluation script: ROUGE-1/2/L + BERTScore.

Run:
    # Zero-shot baseline:
    python -m src.eval --n_samples 100 --output ./outputs/eval_baseline.json

    # Fine-tuned model:
    python -m src.eval --adapter_dir ./outputs/llama32-samsum-lora --n_samples 100 \\
        --output ./outputs/eval_finetuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """ROUGE-1/2/L (rouge-score) + BERTScore-F1.

    We deliberately use rouge-score (the same implementation the SAMSum
    paper and most academic work uses) rather than HF evaluate's rouge,
    which has minor tokenization differences.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for pred, ref in zip(predictions, references, strict=True):
        scores = scorer.score(ref, pred)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
    n = len(predictions)

    from bert_score import score as bert_score

    _, _, f1 = bert_score(predictions, references, lang="en", verbose=False, rescale_with_baseline=True)
    bertscore_f1 = float(f1.mean().item())

    return {
        "rouge1_f": r1 / n,
        "rouge2_f": r2 / n,
        "rougeL_f": rl / n,
        "bertscore_f1": bertscore_f1,
        "n_samples": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Path or HF repo of a LoRA adapter. If None, evaluates the base model zero-shot.")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    from src.infer import batch_summarize
    from src.model import load_for_inference

    print(f"Loading model (adapter={args.adapter_dir})...")
    model, tokenizer = load_for_inference(adapter_repo=args.adapter_dir, merge=True)

    print("Loading SAMSum test split...")
    test_ds = load_dataset("samsum", split=f"test[:{args.n_samples}]", trust_remote_code=True)
    dialogues = list(test_ds["dialogue"])
    references = list(test_ds["summary"])

    print("Generating predictions...")
    predictions = batch_summarize(model, tokenizer, dialogues, batch_size=args.batch_size)

    print("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    print(json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "adapter_dir": args.adapter_dir,
                "metrics": metrics,
                "samples": [
                    {"dialogue": d, "reference": r, "prediction": p}
                    for d, r, p in zip(dialogues[:5], references[:5], predictions[:5], strict=True)
                ],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
