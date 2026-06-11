"""Multi-provider LLM factory.

The rest of the codebase depends only on the LangChain `BaseChatModel`
interface, so providers are interchangeable. Adding a new provider means
adding one branch here — no other file changes.

Why this matters: in interviews you can demonstrate "I built this to be
provider-agnostic so we can swap from the free HF Inference API in
development to a paid managed model in production without code changes."
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chat_llm() -> BaseChatModel:
    provider = settings.llm_provider
    log.info("initializing_llm", provider=provider)

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        if not settings.huggingfacehub_api_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is required for huggingface provider")

        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model_id,
            huggingfacehub_api_token=settings.huggingfacehub_api_token,
            max_new_tokens=settings.max_tokens,
            temperature=settings.temperature,
            task="text-generation",
        )
        return ChatHuggingFace(llm=endpoint)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
