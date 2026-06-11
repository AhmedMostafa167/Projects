"""Chunking is deterministic and doesn't need network — good first test."""

from langchain_core.documents import Document

from src.ingestion import chunk_documents


def test_chunk_documents_splits_long_text():
    long_text = "Sentence. " * 500
    doc = Document(page_content=long_text, metadata={"source": "test"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1200 for c in chunks)
    assert all(c.metadata["source"] == "test" for c in chunks)


def test_chunk_documents_keeps_short_text_in_one_chunk():
    doc = Document(page_content="A short document.", metadata={})
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
