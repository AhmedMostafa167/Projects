"""Cross-encoder reranker.

Bi-encoder retrieval (dense + BM25) is fast but coarse; a cross-encoder
scores each (query, document) pair jointly and reorders the candidates,
which materially improves precision@k at the top of the list.

We load the model lazily so the app boots quickly when reranking isn't
used (e.g. in unit tests).
"""

from functools import lru_cache

from langchain_core.documents import Document

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    # Lazy import so the module is importable in environments where the
    # heavy ML dependency isn't installed (e.g. unit tests that stub it out).
    from sentence_transformers import CrossEncoder

    log.info("loading_reranker", model=settings.reranker_model)
    return CrossEncoder(settings.reranker_model, device="cpu")


class CrossEncoderReranker:
    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.top_k_rerank

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        model = _load_model()
        pairs = [(query, d.page_content) for d in documents]
        scores = model.predict(pairs)
        scored = sorted(zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[: self.top_k]]
