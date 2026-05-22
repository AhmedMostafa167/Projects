# Interview talking points — Banking77 + ModernBERT

Study sheet. Likely questions, 30-second spoken answer, deeper context for follow-ups.

---

## 1. "Walk me through the project."

> I fine-tuned ModernBERT-base — the 2024 successor to BERT — on Banking77, which is a 77-class customer-intent classification benchmark. Full fine-tune, not LoRA, because the model is only 149M params and full fine-tuning is more accurate on small classification datasets. I split a stratified validation set out of train, picked the best checkpoint by macro F1, then ran the final test eval and fit a temperature for calibration. The Gradio demo shows the top-5 predicted intents with calibrated confidence bars and is deployed to Hugging Face Spaces.

---

## 2. "Why ModernBERT and not vanilla BERT or DeBERTa-v3?"

> ModernBERT (Answer.AI, late 2024) keeps the same param budget as BERT-base but ships three upgrades — rotary position embeddings, GeGLU activations, and alternating local/global attention — plus it was trained on 2 trillion tokens of modern web and code data. The net effect is materially better representations and 8k context support. For Banking77 the long context doesn't get used, but the better representations consistently lift macro F1 by 2–4 points over BERT-base in published benchmarks.

---

## 3. "Why full fine-tune here when you used LoRA in Project 2?"

> Two different regimes. Project 2 is generative — decoder-only, 3B parameters, instruction tuning. There LoRA is the right tool: it cuts memory ~4× and trains under 1% of params with negligible quality loss. This project is discriminative — 149M params, 13k training examples, a fresh classification head on top. At that scale full fine-tune is more accurate, faster to converge, and the memory savings of LoRA don't matter because the whole thing fits on a free Colab T4. Picking the right method for the regime is half the job.

---

## 4. "How did you split the data?"

> Banking77 ships with train and test splits. I carved a 10% stratified-by-label validation split out of train, so each of the 77 classes is represented in val. The test split was held out for final reporting — only touched once, after the model and the temperature were both frozen. That's important because if you use the test set for any tuning decisions, the test score is no longer honest.

---

## 5. "Why macro F1 and not accuracy?"

> With 77 classes, class frequencies vary — some intents have hundreds of examples, others have a few dozen. Plain accuracy is dominated by the head classes; if the model is great on common intents and terrible on rare ones, accuracy still looks fine. Macro F1 averages F1 across classes giving each one equal weight, so it actually reflects "did the model learn 77 things, or did it learn the 10 most common things and ignore the rest". I report both, but I pick the best checkpoint on macro F1.

---

## 6. "What is temperature scaling and why do you do it?"

> A neural classifier's softmax confidences are usually overconfident — when the model says "97% sure", it's right less often than 97% of the time. Temperature scaling fits a single positive scalar T such that calibrated probabilities are softmax(logits / T). It doesn't change predictions, only their confidence. I fit T by minimizing NLL on the held-out validation set. Why it matters: if any downstream system thresholds on confidence — like "route to a human if confidence below 0.6" — the threshold is meaningless without calibration. ECE (expected calibration error) is the metric I report to show the gap shrinks after scaling.

**Follow-up to be ready for:**
- *Why not Platt scaling or isotonic regression?* Both are stronger but require more parameters / more validation data. Temperature scaling is one scalar and works well empirically. It's the go-to first choice (Guo et al. 2017).
- *Where did you fit T?* On the validation set, after picking the best checkpoint. Never on test.

---

## 7. "What's `label_smoothing_factor=0.1` doing?"

> Label smoothing replaces the one-hot target [0,...,1,...,0] with a soft target where the correct class gets `1 - ε` and the remaining mass `ε` is spread uniformly across the other classes. With 77 classes and ε=0.1, each wrong class gets ~0.0013. Two effects: it regularizes the model away from making extreme predictions, which improves calibration; and it tends to give a small bump in test accuracy (~0.5–1 point) on classification tasks. It also pairs naturally with temperature scaling because both push against overconfidence.

---

## 8. "Walk me through your training config."

> Adam-W with weight decay 0.01, learning rate 5e-5 with a cosine schedule and 10% warmup, batch size 32 with bf16 mixed precision, 4 epochs, label-smoothing 0.1. Standard recipe for BERT-family fine-tuning. I monitor macro F1 on the val set every epoch and load the best checkpoint at end. I deliberately don't run a hyperparameter sweep — defaults are known-good for this combination, and a sweep would marginally improve F1 without changing the project's story.

---

## 9. "How would you scale this?"

> Two axes. For more classes (say, thousands), I'd switch from a single softmax head to a contrastive / metric-learning setup — embed the query, embed the intent labels, retrieve the closest. That scales as O(num_classes) for inference instead of growing the parameter count linearly. For more data, the recipe stays similar but I'd revisit batch size, add gradient accumulation, and probably move to distributed training with `accelerate`. For lower latency in production, I'd export the model to ONNX and serve with Triton or vLLM.

---

## 10. "What are the failure modes you found?"

> The classic ones for fine-grained intent classification: semantically near-duplicate intents. Banking77 has, for instance, both `card_payment_fee_charged` and `cash_withdrawal_charge` — the model sometimes confuses these because the linguistic signal is similar. I made the demo surface top-5 instead of just top-1 partly for this reason: in a real system, you'd route to a human or to a clarification flow when the second-place probability is close to the first.

---

## Quick facts to memorize

- **Base model**: `answerdotai/ModernBERT-base` (149M, encoder-only, 8k ctx, RoPE, GeGLU, alternating attention)
- **Dataset**: `PolyAI/banking77` (13k train / 3k test, 77 classes)
- **Stratified val split**: 10% of train
- **Optimizer**: AdamW, lr 5e-5, cosine, warmup 10%, weight-decay 0.01
- **Other**: bf16, label-smoothing 0.1, batch 32, 4 epochs
- **Selection metric**: macro F1 (val)
- **Calibration**: temperature scaling, T fit by NLL on val (ternary search)
- **Test metrics reported**: accuracy, macro F1, top-3 acc, top-5 acc, ECE before/after T

---

## If they ask "show me one specific design decision"

Open `src/calibration.py`. Walk through `fit_temperature` (ternary search on NLL), `apply_temperature`, and `expected_calibration_error`. Three short functions, no PyTorch dependency, fully unit-tested. The argument to make: "calibration is a small amount of code that turns the model's confidence numbers from decorative into actually usable downstream."
