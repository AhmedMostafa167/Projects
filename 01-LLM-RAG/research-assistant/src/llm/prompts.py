"""All prompt strings for the LangGraph workflow nodes.

Centralised here so prompt-engineering iterations are reviewable in one place.
Stored as plain strings rather than ChatPromptTemplate since all nodes use
with_structured_output and build their own message lists directly.
"""

REWRITE_QUERY_PROMPT = (
    "You are a research librarian. Rewrite the user's question into 2-3 "
    "concise search queries that will retrieve the most relevant academic "
    "papers and articles."
)

GRADE_PROMPT = (
    "Decide whether the retrieved document is relevant to the question. "
    "Return True if relevant, False if not."
)

GENERATE_PROMPT = (
    "You are a careful research assistant. Answer the user's question using ONLY "
    "the provided context. Cite sources inline as [1], [2], etc. matching the "
    "numbered context blocks. If the context is insufficient, say so explicitly "
    "rather than guessing.\n\n"
    "Context:\n{context}"
)

REFLECT_PROMPT = (
    "Review the draft answer. Assess whether it fully addresses the question "
    "with well-grounded citations. If it does, mark it sufficient. Otherwise, "
    "mark it insufficient and provide ONE focused follow-up search query."
)
