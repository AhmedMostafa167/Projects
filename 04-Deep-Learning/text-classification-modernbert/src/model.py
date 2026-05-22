"""Model + tokenizer loaders.

`load_for_training()` returns a fresh classification head on top of the
base encoder. `load_for_inference()` loads a previously trained model
(local dir or HF Hub repo).
"""

from __future__ import annotations

from src.config import settings


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        settings.base_model_id,
        token=settings.huggingfacehub_api_token,
    )


def load_for_training(num_labels: int, id2label: dict[int, str]):
    """Fresh classification head over the frozen embeddings + transformer."""
    from transformers import AutoModelForSequenceClassification

    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.base_model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        token=settings.huggingfacehub_api_token,
    )
    return model, load_tokenizer()


def load_for_inference(model_repo: str):
    """Load a previously trained model + its tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        model_repo,
        token=settings.huggingfacehub_api_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_repo,
        token=settings.huggingfacehub_api_token,
    )
    model.eval()
    return model, tokenizer
