from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["retrieval-augmented generation"])
    arxiv_n: int = Field(5, ge=0, le=20)
    web_n: int = Field(5, ge=0, le=20)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["What is RAG and why does it help LLMs?"])


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]