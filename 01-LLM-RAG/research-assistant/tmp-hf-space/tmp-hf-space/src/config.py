"""Centralized configuration loaded from environment variables.

Using pydantic-settings means every config value is type-checked at startup,
not at first use — a misconfigured env var fails loudly instead of silently
producing a broken request hours later.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["huggingface", "groq", "anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: ProviderName = "groq"

    huggingfacehub_api_token: str | None = None
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    hf_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openai_model: str = "gpt-4o-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chroma_persist_dir: Path = Path("./data/chroma")
    collection_name: str = "research_papers"

    top_k_dense: int = 10
    top_k_bm25: int = 10
    top_k_rerank: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120

    max_tokens: int = 1024
    temperature: float = 0.2

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "research-assistant"
    log_level: str = "INFO"

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    @property
    def active_api_key(self) -> str | None:
        return {
            "huggingface": self.huggingfacehub_api_token,
            "groq": self.groq_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }[self.llm_provider]


settings = Settings()
