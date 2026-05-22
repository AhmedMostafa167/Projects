"""Evaluation: accuracy, macro F1, top-K accuracy, ECE, confusion matrix.

Run:
    python -m src.eval --model_dir ./outputs/modernbert-banking77 \\
        --output ./outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def top_k_accuracy(logits, labels, k: int) -> float:
    import numpy as np

    top_k = np.argsort(-logits, axis=1)[:, :k]
    correct = (top_k == labels[:, None]).any(axis=1)
    return float(correct.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import DataCollatorWithPadding

    from src.calibration import apply_temperature, expected_calibration_error, fit_temperature
    from src.data import label_names, load_banking77_splits, tokenize_dataset
    from src.model import load_for_inference

    print(f"Loading model from {args.model_dir}...")
    model, tokenizer = load_for_inference(args.model_dir)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    print("Loading data...")
    splits = load_banking77_splits()
    val_tok = tokenize_dataset(splits["validation"], tokenizer)
    test_tok = tokenize_dataset(splits["test"], tokenizer)
    names = label_names()

    collator = DataCollatorWithPadding(tokenizer)

    def gather_logits(ds) -> tuple:
        all_logits = []
        all_labels = []
        ds_iter = ds.with_format("torch")
        for i in range(0, len(ds_iter), args.batch_size):
            batch = [ds_iter[j] for j in range(i, min(i + args.batch_size, len(ds_iter)))]
            inputs = collator(batch)
            labels = inputs.pop("labels").numpy()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model(**inputs)
            all_logits.append(out.logits.float().cpu().numpy())
            all_labels.append(labels)
        return np.concatenate(all_logits), np.concatenate(all_labels)

    print("Scoring validation set (for temperature fitting)...")
    val_logits, val_labels = gather_logits(val_tok)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"  Fitted temperature T = {temperature:.4f}")

    print("Scoring test set...")
    test_logits, test_labels = gather_logits(test_tok)
    preds = test_logits.argmax(axis=1)

    raw_probs = apply_temperature(test_logits, temperature=1.0)
    cal_probs = apply_temperature(test_logits, temperature=temperature)

    metrics = {
        "accuracy": accuracy_score(test_labels, preds),
        "f1_macro": f1_score(test_labels, preds, average="macro"),
        "top_3_accuracy": top_k_accuracy(test_logits, test_labels, k=3),
        "top_5_accuracy": top_k_accuracy(test_logits, test_labels, k=5),
        "ece_uncalibrated": expected_calibration_error(raw_probs, test_labels),
        "ece_calibrated": expected_calibration_error(cal_probs, test_labels),
        "temperature": temperature,
        "n_test": len(test_labels),
    }

    print(json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "metrics": metrics,
                "per_class_report": classification_report(
                    test_labels, preds, target_names=names, output_dict=True, zero_division=0
                ),
            },
            f,
            indent=2,
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
