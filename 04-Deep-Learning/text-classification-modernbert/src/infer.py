"""Inference helper that returns top-K predictions with confidence.

Designed to be the single function the Gradio app calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediction:
    label: str
    confidence: float


def predict_top_k(
    model,
    tokenizer,
    text: str,
    k: int = 5,
    temperature: float = 1.0,
) -> list[Prediction]:
    import numpy as np
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs)

    logits = out.logits.float().cpu().numpy()[0] / temperature
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()

    top_idx = np.argsort(-probs)[:k]
    id2label = model.config.id2label
    return [Prediction(label=id2label[int(i)], confidence=float(probs[i])) for i in top_idx]
