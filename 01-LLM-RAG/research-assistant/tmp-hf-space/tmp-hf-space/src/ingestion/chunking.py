"""Recursive character splitting with sensible defaults for academic text.

Chunk size of 800 / overlap 120 balances:
- enough context per chunk for the model to ground an answer
- small enough that multiple chunks fit in the LLM's context window
- overlap prevents losing facts that span chunk boundaries
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
