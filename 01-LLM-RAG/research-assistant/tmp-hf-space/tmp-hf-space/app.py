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
