"""Open-web search via DuckDuckGo. Returns Documents with source URLs
for citation tracking."""

from duckduckgo_search import DDGS
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import get_logger

log = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def load_web(query: str, max_results: int = 5) -> list[Document]:
    log.info("web_search", query=query, max_results=max_results)

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    docs = [
        Document(
            page_content=f"{r.get('title', '')}\n\n{r.get('body', '')}",
            metadata={
                "source": "web",
                "title": r.get("title", ""),
                "url": r.get("href", ""),
            },
        )
        for r in results
    ]
    log.info("web_loaded", count=len(docs))
    return docs
