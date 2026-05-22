"""Model + tokenizer + LoRA loading.

Two entry points:
    - load_for_training()  -> 4-bit base model with LoRA adapters attached
    - load_for_inference(adapter_repo)  -> base model with the trained adapter
                                            optionally merged for faster gen
"""

from __future__ import annotations

import torch

from src.config import settings


def _bnb_config():
    """4-bit quantization config (QLoRA recipe).

    NF4 + double-quant + bf16 compute is the standard QLoRA preset and
    works well on T4 / A100 / consumer GPUs.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings.base_model_id, token=settings.huggingfacehub_api_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_for_training():
    """Returns (model, tokenizer) with LoRA attached, ready for SFTTrainer."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    quant_config = _bnb_config() if settings.use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=settings.huggingfacehub_api_token,
    )
    model.config.use_cache = False  # required for gradient checkpointing during training

    if settings.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.lora_target_modules_list,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, load_tokenizer()


def load_for_inference(adapter_repo: str | None = None, merge: bool = False):
    """Load base model + optional adapter for generation.

    If `adapter_repo` is None, returns the base model only — useful as a
    zero-shot baseline. If `merge=True`, merges LoRA weights into the
    base for faster inference at the cost of memory.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=settings.huggingfacehub_api_token,
    )
    model.eval()

    if adapter_repo:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_repo, token=settings.huggingfacehub_api_token)
        if merge:
            model = model.merge_and_unload()
            model.eval()

    return model, load_tokenizer()
