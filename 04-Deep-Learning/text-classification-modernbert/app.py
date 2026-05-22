"""Gradio demo: paste a banking query, see top-5 intent predictions
with confidence bars. Falls back to the base ModernBERT (random head)
if no fine-tuned model is configured — keeps the demo runnable even
before training is done.
"""

import os

import gradio as gr

from src.config import settings
from src.infer import predict_top_k
from src.model import load_for_inference

MODEL_REPO = os.getenv("HF_MODEL_REPO") or settings.hf_model_repo or settings.base_model_id

print(f"Loading model from {MODEL_REPO}...")
model, tokenizer = load_for_inference(MODEL_REPO)
print(f"Ready. Number of labels: {model.config.num_labels}")


EXAMPLES = [
    ["My card hasn't arrived yet, what should I do?"],
    ["Why was I charged a fee for using my card?"],
    ["I sent money to the wrong person, can I cancel?"],
    ["How do I verify my identity?"],
    ["My recent transaction is showing as pending"],
]


def predict(text: str) -> tuple[str, dict]:
    if not text.strip():
        return "Please enter a banking query.", {}
    preds = predict_top_k(model, tokenizer, text.strip(), k=5)
    top = preds[0]
    headline = f"**Top intent**: `{top.label}` ({top.confidence:.1%} confidence)"
    bar_data = {p.label: p.confidence for p in preds}
    return headline, bar_data


with gr.Blocks(title="Banking Intent Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Banking Intent Classifier\n"
        f"**Model**: `{MODEL_REPO}`\n\n"
        "Paste a customer-service-style banking query. The model returns the top 5 "
        "predicted intents from the 77-class Banking77 taxonomy."
    )
    inp = gr.Textbox(label="Customer query", lines=2, placeholder="My card hasn't arrived...")
    btn = gr.Button("Classify", variant="primary")
    out_md = gr.Markdown()
    out_chart = gr.Label(num_top_classes=5, label="Top 5 intents (calibrated probabilities)")
    btn.click(predict, inputs=inp, outputs=[out_md, out_chart])
    gr.Examples(examples=EXAMPLES, inputs=inp)
    gr.Markdown(
        "_Fine-tuned ModernBERT on PolyAI/banking77. "
        "[Source on GitHub](https://github.com/AhmedMostafa167/Projects/tree/main/04-Deep-Learning/text-classification-modernbert)._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
