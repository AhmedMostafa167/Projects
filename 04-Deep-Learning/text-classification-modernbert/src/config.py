"""Typed configuration. Defaults are the recipe used to fine-tune
ModernBERT-base on Banking77; everything is overridable via env / .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    huggingfacehub_api_token: str | None = None

    base_model_id: str = "answerdotai/ModernBERT-base"

    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 4
    per_device_batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 128
    label_smoothing: float = 0.1

    # Where the trained model lives after push_to_hub.py
    hf_model_repo: str | None = None


settings = Settings()
