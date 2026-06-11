"""Top-level pipeline: ingest -> index -> answer.

This is what the API and UI call. Encapsulating the orchestration here
keeps the entry points (api.py, app.py) thin.
"""

from dataclasses import dataclass
from typing import Any

from src.graph import build_graph
from src.ingestion import chunk_documents, load_arxiv, load_web
from src.retrieval import HybridRetriever, build_vector_store
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict[str, Any]]


class ResearchPipeline:
    def __init__(self) -> None:
        self.vector_store = build_vector_store()
        self._cached_docs: list = []
        self._graph = None

    def ingest(self, query: str, arxiv_n: int = 5, web_n: int = 5) -> int:
        docs = load_arxiv(query, max_results=arxiv_n) + load_web(query, max_results=web_n)
        chunks = chunk_documents(docs)
        self.vector_store.add_documents(chunks)
        self._cached_docs.extend(chunks)
        # Invalidate the graph so the next ask() rebuilds BM25 over the new corpus.
        self._graph = None
        log.info("ingested", chunks=len(chunks))
        return len(chunks)

    def _ensure_graph(self):
        if self._graph is None:
            retriever = HybridRetriever(self.vector_store, self._cached_docs)
            self._graph = build_graph(retriever)
        return self._graph

    def ask(self, question: str) -> AnswerResult:
        graph = self._ensure_graph()
        final_state = graph.invoke({"question": question})
        sources = [
            {
                "title": d.metadata.get("title", ""),
                "url": d.metadata.get("url", ""),
                "source": d.metadata.get("source", ""),
            }
            for d in final_state.get("documents", [])
        ]
        return AnswerResult(answer=final_state.get("answer", ""), sources=sources)
