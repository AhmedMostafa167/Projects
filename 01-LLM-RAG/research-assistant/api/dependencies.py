from fastapi import Request

from src.pipeline import ResearchPipeline

def get_pipeline(request: Request) -> ResearchPipeline:
    return request.app.state.pipeline