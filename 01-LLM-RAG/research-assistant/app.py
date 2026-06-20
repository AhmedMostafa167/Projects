"""Gradio frontend. This is the entry point for the Hugging Face Space.

Calls the FastAPI service over HTTP rather than importing the pipeline
directly, so there is a single process holding the models and the
vector store. Set API_BASE_URL to point at a different host if the
API is not running on localhost:8000 (e.g. in docker-compose, set it
to http://api:8000).

Run locally:
    uvicorn api.main:app --reload          # terminal 1
    python app.py                          # terminal 2
"""

import os

import gradio as gr
import httpx

from src.utils import get_logger

log = get_logger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def ingest_handler(query: str, arxiv_n: int, web_n: int) -> str:
    if not query.strip():
        return "Please enter a topic to ingest."
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(
                f"{API_BASE_URL}/ingest",
                json={"query": query.strip(), "arxiv_n": int(arxiv_n), "web_n": int(web_n)},
            )
            resp.raise_for_status()
            count = resp.json()["chunks_indexed"]
    except httpx.HTTPStatusError as e:
        log.error("ingest_failed", status=e.response.status_code, body=e.response.text)
        return f"Ingest failed: {e.response.status_code} — {e.response.text}"
    except httpx.RequestError as e:
        log.error("ingest_unreachable", error=str(e))
        return f"Could not reach the API at {API_BASE_URL}: {e}"
    return f"Indexed {count} chunks for topic: {query}"


def ask_handler(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{API_BASE_URL}/ask", json={"question": question.strip()})
            resp.raise_for_status()
            result = resp.json()
    except httpx.HTTPStatusError as e:
        log.error("ask_failed", status=e.response.status_code, body=e.response.text)
        return f"Ask failed: {e.response.status_code} — {e.response.text}", ""
    except httpx.RequestError as e:
        log.error("ask_unreachable", error=str(e))
        return f"Could not reach the API at {API_BASE_URL}: {e}", ""

    sources_md = "\n".join(
        f"- [{s['title'] or s['url']}]({s['url']}) ({s['source']})"
        for s in result["sources"]
        if s.get("url") or s.get("title")
    )
    return result["answer"], sources_md or "_No sources returned._"


with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Research Assistant\n"
        "RAG over ArXiv + open web with LangGraph. "
        "Ingest a topic first, then ask grounded questions."
    )

    with gr.Tab("1. Ingest a topic"):
        topic = gr.Textbox(label="Topic", placeholder="e.g. retrieval-augmented generation")
        with gr.Row():
            arxiv_n = gr.Slider(0, 20, value=5, step=1, label="ArXiv results")
            web_n = gr.Slider(0, 20, value=5, step=1, label="Web results")
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_out = gr.Markdown()
        ingest_btn.click(ingest_handler, inputs=[topic, arxiv_n, web_n], outputs=ingest_out)

    with gr.Tab("2. Ask a question"):
        question = gr.Textbox(label="Question", lines=2)
        ask_btn = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Sources")
        ask_btn.click(ask_handler, inputs=question, outputs=[answer, sources])

    gr.Markdown(
        "_Built with LangChain 0.3 + LangGraph. "
        "[Source on GitHub](https://github.com/AhmedMostafa167/Projects)._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
"""Gradio frontend. This is the entry point for the Hugging Face Space.

Run locally: `python app.py`
"""

import gradio as gr

from src.pipeline import ResearchPipeline
from src.utils import get_logger

log = get_logger(__name__)

pipeline = ResearchPipeline()


def ingest_handler(query: str, arxiv_n: int, web_n: int) -> str:
    if not query.strip():
        return "Please enter a topic to ingest."
    count = pipeline.ingest(query.strip(), arxiv_n=int(arxiv_n), web_n=int(web_n))
    return f"Indexed {count} chunks for topic: {query}"


def ask_handler(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question.", ""
    result = pipeline.ask(question.strip())
    sources_md = "\n".join(
        f"- [{s['title'] or s['url']}]({s['url']}) ({s['source']})"
        for s in result.sources
        if s.get("url") or s.get("title")
    )
    return result.answer, sources_md or "_No sources returned._"


with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Research Assistant\n"
        "RAG over ArXiv + open web with LangGraph. "
        "Ingest a topic first, then ask grounded questions."
    )

    with gr.Tab("1. Ingest a topic"):
        topic = gr.Textbox(label="Topic", placeholder="e.g. retrieval-augmented generation")
        with gr.Row():
            arxiv_n = gr.Slider(0, 20, value=5, step=1, label="ArXiv results")
            web_n = gr.Slider(0, 20, value=5, step=1, label="Web results")
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_out = gr.Markdown()
        ingest_btn.click(ingest_handler, inputs=[topic, arxiv_n, web_n], outputs=ingest_out)

    with gr.Tab("2. Ask a question"):
        question = gr.Textbox(label="Question", lines=2)
        ask_btn = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Sources")
        ask_btn.click(ask_handler, inputs=question, outputs=[answer, sources])

    gr.Markdown(
        "_Built with LangChain 0.3 + LangGraph. "
        "[Source on GitHub](https://github.com/AhmedMostafa167/Projects)._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)