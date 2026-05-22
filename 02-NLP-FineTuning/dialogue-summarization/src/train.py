"""Training entry point.

Run:
    python -m src.train --output_dir ./outputs/llama32-samsum-lora

This is identical to what the training notebook executes under the hood.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    args = parser.parse_args()

    # Heavy imports deferred so `--help` works without them installed.
    from trl import SFTConfig, SFTTrainer

    from src.config import settings
    from src.data import load_samsum
    from src.model import load_for_training

    print(f"Loading {settings.base_model_id} for training...")
    model, tokenizer = load_for_training()

    print("Loading SAMSum...")
    ds = load_samsum()
    train_ds, eval_ds = ds["train"], ds["validation"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=settings.num_epochs,
        per_device_train_batch_size=settings.per_device_batch_size,
        per_device_eval_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_seq_length=settings.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",  # set to "wandb" if you want experiment tracking
        # SFT-specific:
        packing=False,
        assistant_only_loss=True,  # only compute loss on assistant turns
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()

    print(f"Saving adapter to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("Done.")


if __name__ == "__main__":
    main()
