# Architecture

## What this project is

A LoRA fine-tune of `meta-llama/Llama-3.2-3B-Instruct` on the SAMSum dialogue summarization dataset, packaged with a Gradio inference demo and ready to deploy to a Hugging Face Space.

## Pipeline

```
                ┌────────────┐
                │  SAMSum    │  (HF dataset)
                │  ~14.7k    │
                └─────┬──────┘
                      │ load_samsum() — reformats into chat-template messages
                      ▼
                ┌─────────────────────────┐
                │ Llama-3.2-3B-Instruct   │ ← loaded in 4-bit NF4 (QLoRA)
                │ + LoRA adapters         │   on q_proj, k_proj, v_proj, o_proj
                └─────────┬───────────────┘
                          │ trl.SFTTrainer(assistant_only_loss=True)
                          ▼
                ┌──────────────────────┐
                │  LoRA adapter (~30MB) │
                │  outputs/llama32-...  │
                └──────────┬───────────┘
                           │ scripts/push_to_hub.py
                           ▼
                ┌──────────────────────┐
                │   HF Hub model repo  │  ← inference loads from here
                └──────────┬───────────┘
                           │
                ┌──────────┴───────────┐
                ▼                      ▼
        Gradio demo            Eval (src/eval.py)
        (HF Space)             ROUGE-1/2/L + BERTScore
```

## Why LoRA (and specifically QLoRA)?

**Vanilla LoRA**: instead of updating the full weight matrix `W`, we learn two small low-rank matrices `A` and `B` such that the effective weight is `W + BA`. `B` is initialized to zero, so training starts as a no-op. Rank `r=16` means `A` is `r × d` and `B` is `d × r`; we train `2 × r × d` parameters per layer instead of `d × d`.

**QLoRA**: keep the base model frozen *and* quantized to 4-bit NF4. The LoRA adapters stay in bf16 so they have enough precision for gradient updates. This combination is what makes 3B–7B fine-tuning trivial on a single consumer GPU.

**Concretely for this project**:
- Llama-3.2-3B total params: 3.21B
- Trainable LoRA params (r=16 on q/k/v/o): ~24M (~0.75%)
- Base memory: ~2 GB (4-bit)
- Adapter memory: ~95 MB (bf16)
- Training peak memory: ~10–12 GB on T4 with grad-checkpointing — fits with margin

## Why `SFTTrainer` over `Trainer`?

`Trainer` is general-purpose. `SFTTrainer` (from `trl`) is built for supervised instruction fine-tuning and gives us, for free:
- Automatic chat-template application to `messages`-format datasets.
- **`assistant_only_loss=True`**: the trainer constructs a label mask so cross-entropy is computed only on assistant tokens. Without this we'd be partially graded on memorizing the system prompt and user message — wasted signal that degrades sample efficiency.
- `packing` (optional): concatenates short examples into single sequences for throughput. We leave it off here because the loss mask is more reliable with one example per sequence, and SAMSum dialogues are long enough that packing doesn't help much anyway.

## The chat-template trick

SAMSum is a `{dialogue, summary}` pair. We reformat into Llama-3's chat structure:

```
system    → "You are a concise dialogue-summarization assistant..."
user      → "Summarize this conversation:\n\n<dialogue>"
assistant → "<summary>"
```

This matches the format Llama-3 was instruction-tuned on, so the model treats this as a continuation of its existing behavior rather than a domain shift. In practice, this means fine-tuning converges faster and generalizes better than dumping raw text at it.

## Inference path

`src/model.py:load_for_inference` supports three modes:
1. **Base only** (`adapter_repo=None`) — zero-shot baseline for comparison.
2. **Base + adapter** — load adapter on top of base; LoRA matrices remain separate.
3. **Merged** (`merge=True`) — fold LoRA into base for faster generation at the cost of memory. Use this for the production demo.

## Evaluation

`src/eval.py` computes:
- **ROUGE-1/2/L** via `rouge-score` (Google's reference implementation, the one most summarization papers use).
- **BERTScore-F1** via `bert-score` (default RoBERTa-large, rescaled with baseline).

Both metrics live in `src/eval.py:compute_metrics` and can be reused by anyone consuming the package.

## What's not here (and why)

| Missing | Reason |
|---|---|
| Multi-GPU training (DDP/FSDP) | SAMSum + 3B fits on one T4. Adding distributed training would be ceremony without benefit at this scale. |
| DPO / preference fine-tuning | Out of scope — SFT is the right tool for an extractive/abstractive task with a single gold answer per example. |
| Quantized inference (GGUF/AWQ) | Possible but adds packaging complexity. Bf16 + merged LoRA is plenty fast for the Gradio demo. |
| Online evaluation / user feedback | Would matter in production; out of scope for a portfolio piece. |
