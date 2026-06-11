"""LangGraph workflow for the research assistant.

Why LangGraph instead of a plain LCEL chain?
A research-answering task isn't strictly linear: depending on what the
retriever returns we may want to rewrite the query, retrieve more, or
generate. LangGraph models this as a state machine with explicit nodes
and conditional edges — easier to reason about, easier to add nodes,
and built-in support for streaming intermediate state to the UI.

Flow:
    rewrite_query → retrieve → grade → generate → reflect → END
                                                       ↓
                                               (or back to retrieve once)
"""

from operator import add
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.llm import get_chat_llm
from src.llm.prompts import GENERATE_PROMPT, GRADE_PROMPT, REFLECT_PROMPT, REWRITE_QUERY_PROMPT
from src.retrieval import HybridRetriever
from src.utils import get_logger

log = get_logger(__name__)

MAX_REFLECTION_LOOPS = 1


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchQueries(BaseModel):
    search_queries: List[str] = Field(
        description="2-3 concise search queries derived from the user's question."
    )

class GradeDocument(BaseModel):
    relevant: bool = Field(
        description="True if the document is relevant to the question, False otherwise."
    )

class Reflection(BaseModel):
    sufficient: bool = Field(
        description="True if the draft answer fully addresses the question."
    )
    follow_up_query: Optional[str] = Field(
        default=None,
        description="A single focused search query to fill the gap, if insufficient.",
    )


# ── State ─────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    question: str
    search_queries: List[str]
    documents: Annotated[List[Document], add]
    answer: str
    reflection_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {d.metadata.get('title', d.metadata.get('url', 'source'))}\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(retriever: HybridRetriever):
    """Compile and return a runnable LangGraph workflow."""
    llm = get_chat_llm()

    def rewrite_query(state: ResearchState) -> ResearchState:
        structured_llm = llm.with_structured_output(SearchQueries)
        result = structured_llm.invoke([
            SystemMessage(content=REWRITE_QUERY_PROMPT),
            HumanMessage(content=state["question"]),
        ])
        log.info("rewrote_queries", queries=result.search_queries)
        return {"search_queries": result.search_queries or [state["question"]]}

    def retrieve(state: ResearchState) -> ResearchState:
        all_docs: List[Document] = []
        for q in state.get("search_queries", [state["question"]]):
            all_docs.extend(retriever.retrieve(q))
        return {"documents": all_docs}

    def grade(state: ResearchState) -> ResearchState:
        grader = llm.with_structured_output(GradeDocument)
        kept: List[Document] = []
        for doc in state["documents"]:
            result = grader.invoke([
                SystemMessage(content=GRADE_PROMPT),
                HumanMessage(content=f"Question: {state['question']}\n\nDocument:\n{doc.page_content}"),
            ])
            if result.relevant:
                kept.append(doc)
        log.info("graded", kept=len(kept), total=len(state["documents"]))
        return {"documents": kept if kept else state["documents"][:3]}

    def generate(state: ResearchState) -> ResearchState:
        answer = llm.invoke([
            SystemMessage(content=GENERATE_PROMPT.format(context=_format_context(state["documents"]))),
            HumanMessage(content=state["question"]),
        ])
        return {"answer": answer.content}

    def reflect(state: ResearchState) -> ResearchState:
        reflector = llm.with_structured_output(Reflection)
        result = reflector.invoke([
            SystemMessage(content=REFLECT_PROMPT),
            HumanMessage(content=(
                f"Question: {state['question']}\n\n"
                f"Draft answer:\n{state['answer']}"
            )),
        ])
        count = state.get("reflection_count", 0) + 1

        if result.sufficient or not result.follow_up_query or count >= MAX_REFLECTION_LOOPS:
            return {"reflection_count": count}

        return {"reflection_count": count, "search_queries": [result.follow_up_query]}

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
