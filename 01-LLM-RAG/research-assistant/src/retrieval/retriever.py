"""Hybrid retriever: dense (Chroma) + sparse (BM25) -> rerank.

Dense retrieval captures semantic similarity ("transformer architecture"
matches "self-attention mechanism"). BM25 captures exact-keyword matches
that embeddings sometimes miss (rare terms, acronyms, code identifiers).
Combining both, then reranking with a cross-encoder, consistently beats
either method alone.
"""

from langchain_core.documents import Document

from src.config import settings
from src.retrieval.reranker import CrossEncoderReranker
from src.utils import get_logger

log = get_logger(__name__)


class HybridRetriever:
    """Combine dense + BM25 candidates, dedupe, then rerank."""

    def __init__(
        self,
        vector_store,
        documents: list[Document],
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        from langchain_community.retrievers import BM25Retriever

        self.vector_store = vector_store
        self.bm25 = BM25Retriever.from_documents(documents) if documents else None
        if self.bm25:
            self.bm25.k = settings.top_k_bm25
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(self, query: str) -> list[Document]:
        dense = self.vector_store.similarity_search(query, k=settings.top_k_dense)
        sparse = self.bm25.invoke(query) if self.bm25 else []

        # Dedupe by page_content; keep first occurrence.
        seen: set[str] = set()
        merged: list[Document] = []
        for doc in dense + sparse:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        log.info(
            "retrieved",
            dense=len(dense),
            sparse=len(sparse),
            merged=len(merged),
        )
        return self.reranker.rerank(query, merged)
