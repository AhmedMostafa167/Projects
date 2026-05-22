"""Generation utilities. The Gradio app and the eval script both call into here."""

from __future__ import annotations

from src.config import settings
from src.data import build_inference_prompt


def summarize(model, tokenizer, dialogue: str) -> str:
    import torch

    prompt = build_inference_prompt(dialogue, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Slice off the prompt tokens — only return the model's new tokens.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def batch_summarize(model, tokenizer, dialogues: list[str], batch_size: int = 4) -> list[str]:
    """Naive batched generation. Good enough for the test set (~800 rows)."""
    import torch

    tokenizer.padding_side = "left"  # required for batched generation
    results: list[str] = []

    for i in range(0, len(dialogues), batch_size):
        batch = dialogues[i : i + batch_size]
        prompts = [build_inference_prompt(d, tokenizer) for d in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                do_sample=settings.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, out in enumerate(outputs):
            new_tokens = out[inputs["input_ids"].shape[1]:]
            results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return results
