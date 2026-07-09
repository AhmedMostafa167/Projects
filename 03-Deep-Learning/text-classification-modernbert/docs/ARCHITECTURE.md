# Architecture

## What this project is

A full fine-tune of `answerdotai/ModernBERT-base` on `PolyAI/banking77` for 77-class customer-intent classification, with temperature-scaled output probabilities and a Gradio demo that surfaces top-K predictions with confidence bars.

## Pipeline

```
        ┌──────────────────┐
        │ Banking77        │ (HF dataset, 13k train / 3k test, 77 labels)
        └────────┬─────────┘
                 │ stratified split: train (90%) / val (10%) / test (held out)
                 ▼
        ┌──────────────────────┐
        │ ModernBERT-base      │  encoder-only, 149M params, 8k context
        │ + classification head│  fresh linear layer, num_labels=77
        └────────┬─────────────┘
                 │ transformers.Trainer, bf16, cosine LR, label-smoothing=0.1
                 ▼
        ┌──────────────────────┐
        │ Best checkpoint       │  selected by macro F1 on val
        │ (best_model_at_end)   │
        └────────┬─────────────┘
                 │ scripts/push_to_hub.py with auto-generated model card
                 ▼
        ┌──────────────────────┐
        │ HF Hub repo          │
        └────────┬─────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  Gradio demo         src/eval.py
  (top-K +            (acc, F1, top-K,
   confidence)         ECE, calibration)
```

## Why full fine-tune (not LoRA)?

For a 149M-parameter encoder on a small classification task (~13k examples), full fine-tuning is the right tool:
- The compute / memory budget is small enough that LoRA's main benefit (memory) doesn't matter.
- Classification heads are randomly initialized and need to learn from scratch — LoRA on a frozen base + a fresh head is actually *less* efficient than just unfreezing.
- For final-mile classification accuracy, full fine-tuning consistently outperforms LoRA on small datasets.

Project 2 (decoder-only generation, 3B params, multi-task instruction tuning) is the opposite regime — there LoRA is the right tool.

This contrast is deliberate: knowing when to use each method matters more than picking one and applying it everywhere.

## Why ModernBERT?

`answerdotai/ModernBERT-base` (Dec 2024) is the BERT-family update:
- **Rotary position embeddings** instead of learned positional embeddings → better length generalization.
- **GeGLU activations** instead of GELU → marginal improvement, free.
- **Alternating local + global attention** → faster, supports 8k context.
- **Trained on ~2T tokens** of modern web/code text → much stronger representations than original BERT.

For Banking77, the long-context advantage isn't used (queries are short), but the better representations matter: expected macro F1 is ~92–94%, vs ~89–91% for vanilla `bert-base`.

## Why temperature scaling?

A raw softmax confidence is often miscalibrated — when a modern over-parameterized classifier says "97% sure", it might only be right 88% of the time. That gap is *expected calibration error* (ECE).

Temperature scaling fits a single scalar `T > 0` and rescales logits to `logits / T` before softmax. It:
- doesn't change predictions (argmax is invariant under monotonic scaling),
- only re-shapes confidence to match observed accuracy.

I fit `T` by minimizing NLL on the held-out validation set with a 1-D ternary search (50 iterations is plenty for 1-D). This is honest: predictions don't move, but the confidence numbers you display to a user actually mean what they appear to mean.

If a downstream system thresholds on confidence (e.g., "if confidence < 0.6, route to a human"), calibration matters a lot.

## Evaluation choices

| Metric | Why |
|---|---|
| **Accuracy** | The headline number everyone asks about. |
| **Macro F1** | More honest for 77-class with imbalanced label distribution — averages F1 across classes giving each equal weight. |
| **Top-3 / Top-5 accuracy** | For a chatbot routing layer, you might present the top 3 to a human or use them as a fallback ranking. |
| **ECE (15-bin)** | Whether confidence means anything. |

## What's not here (and why)

| Missing | Reason |
|---|---|
| Hyperparameter search | The defaults are conservative and known-good for BERT-family fine-tuning on Banking77. A sweep would marginally improve F1 but not change the project's signal. |
| ONNX export | Could be added for production but not free in complexity — the model serves fine through HF Transformers. |
| Active learning | The labeled training set is fixed (Banking77 is a benchmark). Active learning would matter if I were collecting new data. |
| Multi-language | Banking77 is English-only; I'd swap to `XNLI` or `MASSIVE` for cross-lingual. |
