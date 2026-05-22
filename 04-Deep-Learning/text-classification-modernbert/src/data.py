"""Banking77 dataset loading and tokenization.

Banking77 has a `train`/`test` split out of the box. For honest reporting
we carve a `validation` set from the train split (stratified by label)
and reserve `test` strictly for the final number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, load_dataset

from src.config import settings

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DATASET_ID = "PolyAI/banking77"


def load_banking77_splits(val_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """Returns train / validation / test splits with `text` and `label` columns."""
    raw = load_dataset(DATASET_ID)
    # Stratify so each of the 77 classes is represented in the val split.
    split = raw["train"].train_test_split(
        test_size=val_size,
        seed=seed,
        stratify_by_column="label",
    )
    return DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    )


def label_names() -> list[str]:
    raw = load_dataset(DATASET_ID, split="train")
    return raw.features["label"].names


def tokenize_dataset(
    ds: Dataset,
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int | None = None,
) -> Dataset:
    max_length = max_length or settings.max_seq_length

    def _tok(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding=False,  # let the collator pad to batch max -> faster
            truncation=True,
            max_length=max_length,
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])
