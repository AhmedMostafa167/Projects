"""SAMSum dataset loading and chat-template formatting.

SAMSum example:
    {
        "id": "...",
        "dialogue": "Hannah: Hey, do you have...",
        "summary": "Hannah asked for Larry's number..."
    }

We reformat each row into the Llama-3 chat template so the model trains on
the natural instruction format it was aligned for. Response-only loss
masking (handled by SFTTrainer's `assistant_only_loss=True`) ensures the
model only learns from the summary tokens, not the prompt.
"""

from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a concise dialogue-summarization assistant. "
    "Given a conversation, produce a short, neutral summary in 1-3 sentences."
)


def _format_row(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this conversation:\n\n{row['dialogue']}"},
            {"role": "assistant", "content": row["summary"]},
        ]
    }


def load_samsum(split: str | None = None) -> Dataset | DatasetDict:
    """Load SAMSum and reformat as chat messages."""
    ds = load_dataset("samsum", trust_remote_code=True)
    if split is not None:
        ds = ds[split]
    return ds.map(_format_row, remove_columns=["id", "dialogue", "summary"])


def build_inference_prompt(dialogue: str, tokenizer: "PreTrainedTokenizerBase") -> str:
    """Apply the chat template for inference (no assistant turn — model will generate it)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize this conversation:\n\n{dialogue}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
