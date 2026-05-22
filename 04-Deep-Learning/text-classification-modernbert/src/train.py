"""Training entry point.

Run:
    python -m src.train --output_dir ./outputs/modernbert-banking77

Uses the standard `transformers.Trainer` (not SFTTrainer) because this is
a discriminative classification task, not instruction tuning. Full
fine-tuning of all 149M params — different signal from Project 2's LoRA.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--logging_steps", type=int, default=25)
    args = parser.parse_args()

    # Heavy imports deferred so `--help` works without them installed.
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    from src.config import settings
    from src.data import label_names, load_banking77_splits, tokenize_dataset
    from src.model import load_for_training

    print("Loading Banking77 splits...")
    splits = load_banking77_splits()
    names = label_names()
    id2label = dict(enumerate(names))
    print(f"  train={len(splits['train'])}  val={len(splits['validation'])}  test={len(splits['test'])}")
    print(f"  num_labels={len(names)}")

    print(f"Loading {settings.base_model_id}...")
    model, tokenizer = load_for_training(num_labels=len(names), id2label=id2label)

    print("Tokenizing...")
    train_tok = tokenize_dataset(splits["train"], tokenizer)
    val_tok = tokenize_dataset(splits["validation"], tokenizer)

    collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=settings.num_epochs,
        per_device_train_batch_size=settings.per_device_batch_size,
        per_device_eval_batch_size=settings.per_device_batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type="cosine",
        label_smoothing_factor=settings.label_smoothing,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=args.logging_steps,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print(f"Saving best model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("Done.")


if __name__ == "__main__":
    main()
