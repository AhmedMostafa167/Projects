"""LangGraph workflow for the research assistant.

Why LangGraph instead of a plain LCEL chain?
A research-answering task isn't strictly linear: depending on what the
retriever returns we may want to rewrite the query, retrieve more, or
generate. LangGraph models this as a state machine with explicit nodes
and conditional edges — easier to reason about, easier to add nodes,
and built-in support for streaming intermediate state to the UI.

Flow:
    rewrite_query → retrieve → grade → [generate | rewrite_query]
                                           ↓
                                       reflect → END
                                           ↓
                                     (or back to retrieve once)
"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from src.llm import (
    ANSWER_PROMPT,
    GRADE_PROMPT,
    QUERY_REWRITE_PROMPT,
    REFLECTION_PROMPT,
    get_chat_llm,
)
from src.retrieval import HybridRetriever
from src.utils import get_logger

log = get_logger(__name__)

MAX_REFLECTION_LOOPS = 1


def _append(left: list, right: list) -> list:
    return left + right


class ResearchState(TypedDict, total=False):
    question: str
    search_queries: list[str]
    documents: Annotated[list[Document], _append]
    answer: str
    reflection_count: int


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {d.metadata.get('title', d.metadata.get('url', 'source'))}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def build_graph(retriever: HybridRetriever):
    """Compile and return a runnable LangGraph workflow."""
    llm = get_chat_llm()

    def rewrite_query(state: ResearchState) -> ResearchState:
        chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"question": state["question"]})
        queries = [q.strip() for q in raw.splitlines() if q.strip()]
        log.info("rewrote_queries", queries=queries)
        return {"search_queries": queries[:3] or [state["question"]]}

    def retrieve(state: ResearchState) -> ResearchState:
        all_docs: list[Document] = []
        for q in state.get("search_queries", [state["question"]]):
            all_docs.extend(retriever.retrieve(q))
        return {"documents": all_docs}

    def grade(state: ResearchState) -> ResearchState:
        chain = GRADE_PROMPT | llm | StrOutputParser()
        kept: list[Document] = []
        for doc in state["documents"]:
            verdict = chain.invoke(
                {"question": state["question"], "document": doc.page_content}
            ).strip().lower()
            if verdict.startswith("yes"):
                kept.append(doc)
        log.info("graded", kept=len(kept), total=len(state["documents"]))
        # Replace, not append — we filtered.
        return {"documents": kept if kept else state["documents"][:3]}

    def generate(state: ResearchState) -> ResearchState:
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "question": state["question"],
                "context": _format_context(state["documents"]),
            }
        )
        return {"answer": answer}

    def reflect(state: ResearchState) -> ResearchState:
        chain = REFLECTION_PROMPT | llm | StrOutputParser()
        verdict = chain.invoke(
            {"question": state["question"], "draft": state["answer"]}
        ).strip()
        count = state.get("reflection_count", 0) + 1
        if verdict.lower().startswith("ok") or count >= MAX_REFLECTION_LOOPS:
            return {"reflection_count": count}
        # Use the follow-up query for one more retrieval round.
        return {"reflection_count": count, "search_queries": [verdict]}

    def should_continue(state: ResearchState) -> str:
        if state.get("reflection_count", 0) >= MAX_REFLECTION_LOOPS:
            return END
        if not state.get("search_queries"):
            return END
        return "retrieve"

    workflow = StateGraph(ResearchState)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("generate", generate)
    workflow.add_node("reflect", reflect)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")
    workflow.add_edge("generate", "reflect")
    workflow.add_conditional_edges("reflect", should_continue, {"retrieve": "retrieve", END: END})

    return workflow.compile()
