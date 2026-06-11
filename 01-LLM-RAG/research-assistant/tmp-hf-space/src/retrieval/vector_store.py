"""Chroma vector store with HuggingFace embeddings.

Using `all-MiniLM-L6-v2` (384-dim, ~80MB) because it runs on CPU in a HF
Spaces free tier and gives competitive retrieval quality. For higher
quality at the cost of memory, swap to `bge-large-en-v1.5` (1024-dim).
"""

from functools import lru_cache

from langchain_core.documents import Document

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings():
    # Lazy import so the package is importable for unit tests that don't
    # actually touch the embedding model.
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("loading_embeddings", model=settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(documents: list[Document] | None = None):
    from langchain_chroma import Chroma

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
    )
    if documents:
        log.info("adding_documents", count=len(documents))
        store.add_documents(documents)
    return store
