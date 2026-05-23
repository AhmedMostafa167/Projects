---
title: Research Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG over ArXiv + open web, LangGraph powered
---

# Research Assistant

A RAG system built with LangChain 0.3 + LangGraph.

- Ingests papers from ArXiv and the open web
- Hybrid retrieval (dense + BM25) with cross-encoder reranking
- Multi-step LangGraph workflow with query rewriting and self-reflection
- Multi-provider LLM backend (HF Inference, Groq, Anthropic, OpenAI)

**Source code**: [GitHub](https://github.com/AhmedMostafa167/Projects/tree/main/01-LLM-RAG/research-assistant)
