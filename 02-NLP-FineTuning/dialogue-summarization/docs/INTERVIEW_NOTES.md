# Interview talking points — Dialogue Summarization

This is your study sheet. Each section is a likely question, a 30-second answer to give out loud, and the deeper context for follow-ups.

---

## 1. "Walk me through the project."

> I fine-tuned Llama-3.2-3B-Instruct on the SAMSum dialogue summarization dataset using QLoRA — that's LoRA adapters on top of a 4-bit-quantized base model. The training data is reformatted into Llama-3's chat template and trained with `SFTTrainer` with `assistant_only_loss=True` so the loss is computed only on the summary tokens. I evaluated with both ROUGE and BERTScore against the zero-shot baseline. The adapter is published to the Hugging Face Hub and there's a Gradio Space that loads it for live inference. It's a rebuild of my earlier PEGASUS-based project with modern parameter-efficient fine-tuning.

---

## 2. "Why LoRA and not full fine-tuning?"

> Three reasons. One, parameter efficiency — I train under 1% of the parameters and get nearly the quality of full fine-tuning. Two, memory — combined with 4-bit quantization, a 3B model fits in ~12 GB during training, so I can do this on a free Colab T4 instead of needing an A100. Three, deployability — the adapter is 30 MB instead of multiple gigabytes, so it's easy to ship and version on the Hub. The cost is a small quality gap versus full fine-tuning on some tasks, but for instruction-style summarization that gap is empirically tiny.

**Follow-ups to be ready for:**
- *Why rank 16?* Sweet spot from the LoRA paper and subsequent practice. Lower ranks (4, 8) often suffice for adaptation tasks; higher ranks (32, 64) help for larger domain shifts but bring diminishing returns.
- *Why `q_proj, k_proj, v_proj, o_proj`?* These are the attention projections. The QLoRA paper found that targeting all attention modules gives a consistent improvement over q/v only, with minimal memory cost.
- *What's `alpha` doing?* It's a scaling factor: the effective LoRA update is `(alpha/r) * B*A`. Setting `alpha=2r` is the common default.

---

## 3. "Why a decoder-only model? PEGASUS was encoder-decoder."

> In 2020 encoder-decoder models like PEGASUS were the SOTA for summarization. In 2026 the strong baseline is an instruction-tuned decoder-only LLM, for three reasons. One, modern alignment data is overwhelmingly instruction-format, which decoder-only models consume natively. Two, decoder-only architectures generalize better to new prompt formats — I can change the system prompt and immediately have a slightly different task without retraining. Three, in a real production stack you usually have one LLM serving multiple tasks, and decoder-only models share infrastructure better than mixing architectures.

---

## 4. "Why `assistant_only_loss=True`?"

> Without it, the trainer computes cross-entropy on every token in the sequence, including the system prompt and user message. That means the model is partly being graded on memorizing the prompt — which doesn't help summarization quality and wastes gradient signal. With assistant-only loss, the trainer masks all non-assistant tokens out of the loss computation, so we're learning only "given this conversation, produce this summary," which is exactly the task. It's a small change with a measurable impact on sample efficiency.

---

## 5. "How do you evaluate, and why both ROUGE and BERTScore?"

> ROUGE measures n-gram overlap. It's fast, well-understood, and what every summarization paper reports — but it punishes valid paraphrases. BERTScore uses contextual embeddings to compute per-token similarity, so it credits the model when it says the same thing differently. Reporting both is the modern convention. If they disagree — say, ROUGE drops but BERTScore stays flat — that's a signal the model is paraphrasing more, which may or may not be a problem depending on the use case. For summarization where the reference is one of many valid summaries, BERTScore is the more honest metric.

---

## 6. "What does QLoRA actually do at the matrix level?"

> Take a base weight matrix `W` of shape `(d_out, d_in)`. We freeze it and quantize it to 4-bit NF4. Then we add a learned update `BA` where `B` is `(d_out, r)` and `A` is `(r, d_in)`, both in bf16. At forward time, the effective weight is `dequantize(W) + scale * BA`. `B` is initialized to zero so the initial output is identical to the base model — training is a smooth perturbation from there. Only `A` and `B` get gradients, so the optimizer state is also tiny.

---

## 7. "What's the chat-template stuff for?"

> Llama-3 was instruction-tuned with a specific format — `<|system|>...<|user|>...<|assistant|>...`. If I just dump raw `(dialogue, summary)` text at it, I'm asking it to adapt to a format it doesn't recognize, which costs sample efficiency. By reformatting SAMSum into the same chat-template messages format, fine-tuning becomes a gentle specialization — the model is already in "respond as an assistant" mode, I'm just teaching it to specialize in summaries.

---

## 8. "How would you scale this to a larger model or more data?"

> Two axes. For larger models — say Llama-3.1-70B — I'd switch from QLoRA on a single GPU to FSDP across multi-GPU with QLoRA still on top (`bnb_4bit_quant_storage=bfloat16` to make the quantized weights shardable). For more data, the bottleneck is usually data quality, not training compute, so I'd start by filtering and deduplicating the dataset rather than throwing more examples at it. If I really did have 10× the data, I'd add packing to the SFTTrainer config to improve throughput, and I'd run a hyperparameter sweep on rank and learning rate because LoRA's optimal `r` does increase with dataset size.

---

## 9. "How is this deployed?"

> Two pieces. The adapter weights are pushed to a HF Hub model repo with `scripts/push_to_hub.py`, which also generates a model card. The Gradio demo is deployed to a HF Space (Docker SDK) with `scripts/deploy_hf_space.sh`. The demo loads the base model + the adapter from the Hub on startup, merges them for faster inference, and serves a simple UI. It runs on the free CPU Space tier — slowly, but it runs.

---

## 10. "What did you find hardest / what would you do differently?"

**Pick one to tell as a story:**
- *Tokenizer pad-token issues.* Llama tokenizers don't ship with a `pad_token`, so I had to set it to `eos_token` and switch `padding_side` between training (`right`) and batched inference (`left`). It's the kind of thing that fails silently — your model trains but generates garbage in batched mode.
- *Right gradient-checkpointing setting.* `gradient_checkpointing_kwargs={"use_reentrant": False}` is required for newer `transformers` + bf16 — the reentrant default warns but still produces wrong gradients in some configurations.
- *The eval set is too small.* I used 100 samples to keep iteration fast, but variance on ROUGE at n=100 is non-trivial. In a real project I'd use the full 819-row test split for the final report.

---

## Quick facts to memorize

- **Base model**: `meta-llama/Llama-3.2-3B-Instruct` (3.21B params, decoder-only)
- **Dataset**: SAMSum — 14,732 train / 819 val / 818 test
- **LoRA config**: r=16, alpha=32, dropout=0.05, on q/k/v/o projections
- **Quantization**: 4-bit NF4 + double-quant + bf16 compute (QLoRA)
- **Trainer**: `trl.SFTTrainer`, assistant_only_loss=True, packing=False
- **Optimizer**: AdamW, lr=2e-4, cosine schedule, warmup_ratio=0.03
- **Effective batch size**: 8 (per-device 2 × grad-accum 4)
- **Epochs**: 2
- **Max seq length**: 1024
- **Trainable params**: ~24M (~0.75%)
- **Adapter size on disk**: ~30 MB
- **Eval metrics**: ROUGE-1/2/L (rouge-score), BERTScore-F1

---

## If they ask "show me one specific design decision"

Open `src/data.py` and point at `_format_row`. Explain how the chat template aligns the fine-tuning task with the model's instruction-tuned prior — that's the highest-leverage decision in the whole project, and it's two lines of code.
