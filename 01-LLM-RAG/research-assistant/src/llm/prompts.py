"""Prompt templates for each node in the LangGraph workflow.

Kept in one file so prompt-engineering iterations are reviewable in PRs.
"""

from langchain_core.prompts import ChatPromptTemplate

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research librarian. Rewrite the user's question into 2-3 "
            "concise search queries that will retrieve the most relevant academic "
            "papers and articles. Return only the queries, one per line, no numbering.",
        ),
        ("human", "{question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether the retrieved document is relevant to the question. "
            "Reply with a single word: 'yes' or 'no'.",
        ),
        (
            "human",
            "Question: {question}\n\nDocument:\n{document}\n\nIs this document relevant?",
        ),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful research assistant. Answer the user's question using ONLY "
            "the provided context. Cite sources inline as [1], [2], etc. matching the "
            "numbered context blocks. If the context is insufficient, say so explicitly "
            "rather than guessing.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

REFLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Review the draft answer. If it fully addresses the question with "
            "well-grounded citations, reply with 'ok'. Otherwise, reply with a short "
            "follow-up search query that would fill the gap.",
        ),
        (
            "human",
            "Question: {question}\n\nDraft answer:\n{draft}\n\n"
            "Reply 'ok' or a single follow-up query.",
        ),
    ]
)
