"""Gradio demo. Loads the base model + your fine-tuned adapter from the
Hugging Face Hub if `HF_ADAPTER_REPO` is set, otherwise falls back to the
zero-shot base model so the demo is always runnable.

Entry point for the Hugging Face Space.
"""

import os

import gradio as gr

from src.config import settings
from src.infer import summarize
from src.model import load_for_inference

ADAPTER_REPO = os.getenv("HF_ADAPTER_REPO") or settings.hf_adapter_repo

print(f"Loading model (adapter={ADAPTER_REPO or 'none, zero-shot'})...")
model, tokenizer = load_for_inference(adapter_repo=ADAPTER_REPO, merge=True)
print("Ready.")


EXAMPLES = [
    [
        "Hannah: Hey, do you have Betty's number?\n"
        "Amanda: Lemme check\n"
        "Amanda: Sorry, can't find it.\n"
        "Amanda: Ask Larry\n"
        "Amanda: He called her last time we were at the park together\n"
        "Hannah: I don't know him well\n"
        "Hannah: Better text him 🙂\n"
        "Amanda: Don't be shy, he's very nice\n"
        "Hannah: If you say so..\n"
        "Hannah: I'd rather you texted him\n"
        "Amanda: Just text him 🙂\n"
        "Hannah: Urgh.. Alright\n"
        "Hannah: Bye\n"
        "Amanda: Bye bye"
    ],
    [
        "Eric: MACHINE!\n"
        "Rob: That's so gr8!\n"
        "Eric: I know! And shows how Americans see Russian ;)\n"
        "Rob: And it's really funny!\n"
        "Eric: I know! I especially like the train part!\n"
        "Rob: Hahaha! No one talks to the machine like that!\n"
        "Eric: Is this his only stand-up?\n"
        "Rob: Idk. I'll check.\n"
        "Eric: Sure.\n"
        "Rob: Turns out no! There are like 5 of them. He has his own youtube channel.\n"
        "Eric: Thanks for the info!\n"
        "Rob: Sure :)"
    ],
]


def predict(dialogue: str) -> str:
    if not dialogue.strip():
        return "Please paste a conversation to summarize."
    return summarize(model, tokenizer, dialogue.strip())


with gr.Blocks(title="Dialogue Summarization", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Dialogue Summarization\n"
        f"**Model**: `{settings.base_model_id}`"
        + (f" + LoRA adapter `{ADAPTER_REPO}`" if ADAPTER_REPO else " (zero-shot baseline)")
        + "\n\nPaste a conversation and get a 1–3 sentence summary."
    )
    inp = gr.Textbox(label="Conversation", lines=15, placeholder="Speaker A: ...\nSpeaker B: ...")
    out = gr.Textbox(label="Summary", lines=4)
    btn = gr.Button("Summarize", variant="primary")
    btn.click(predict, inputs=inp, outputs=out)
    gr.Examples(examples=EXAMPLES, inputs=inp)
    gr.Markdown(
        "_Fine-tuned with LoRA on SAMSum. "
        "[Source on GitHub](https://github.com/AhmedMostafa167/Projects/tree/main/02-NLP-FineTuning/dialogue-summarization)._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
