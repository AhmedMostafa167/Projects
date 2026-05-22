"""Typed configuration. Every hyperparameter has a default and can be
overridden via env var or .env file. No magic constants scattered across
training and inference code."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    huggingfacehub_api_token: str | None = None

    base_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 2
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    use_4bit: bool = True

    # Inference
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.9

    # Where the adapter lives after training
    hf_adapter_repo: str | None = None

    @property
    def lora_target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.lora_target_modules.split(",") if m.strip()]


settings = Settings()
