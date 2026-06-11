"""Load papers from ArXiv. Returns LangChain Documents with rich metadata
(title, authors, published date, arxiv_id) so they can be used for citations."""

import arxiv
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import get_logger

log = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_arxiv(query: str, max_results: int = 5) -> list[Document]:
    """Search ArXiv and return summaries as Documents.

    We use abstracts (not full PDFs) for speed — full-PDF ingestion is
    available via `pypdf` if a paper turns out to be highly relevant.
    """
    log.info("arxiv_search", query=query, max_results=max_results)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()

    docs: list[Document] = []
    for result in client.results(search):
        docs.append(
            Document(
                page_content=f"{result.title}\n\n{result.summary}",
                metadata={
                    "source": "arxiv",
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": ", ".join(a.name for a in result.authors),
                    "published": result.published.isoformat(),
                    "url": result.entry_id,
                },
            )
        )
    log.info("arxiv_loaded", count=len(docs))
    return docs
