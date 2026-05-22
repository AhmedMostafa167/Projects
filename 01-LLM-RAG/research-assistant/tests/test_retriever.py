"""Hybrid retriever logic with a stubbed reranker — no models loaded."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.retrieval.retriever import HybridRetriever


class _StubReranker:
    def rerank(self, query, documents):
        return documents


def test_hybrid_retriever_dedupes_overlapping_results():
    shared = Document(page_content="shared content " * 30, metadata={"id": 1})
    dense_only = Document(page_content="dense only " * 30, metadata={"id": 2})
    sparse_only = Document(page_content="sparse only " * 30, metadata={"id": 3})

    vs = MagicMock()
    vs.similarity_search.return_value = [shared, dense_only]

    retriever = HybridRetriever(
        vector_store=vs,
        documents=[shared, sparse_only],
        reranker=_StubReranker(),
    )
    results = retriever.retrieve("query")
    contents = [d.page_content for d in results]
    assert contents.count(shared.page_content) == 1
    assert dense_only.page_content in contents
    assert sparse_only.page_content in contents
