"""Pydantic models used by the API."""

from pydantic import BaseModel, Field, field_validator


class IngestRequest(BaseModel):
    """Documents submitted for indexing."""

    documents: list[str] = Field(min_length=1)


class IngestResponse(BaseModel):
    """Summary of a completed indexing request."""

    indexed_documents: int
    indexed_chunks: int


class QueryRequest(BaseModel):
    """Question submitted to the RAG endpoint."""

    question: str

    @field_validator("question")
    @classmethod
    def require_nonempty_question(cls, value: str) -> str:
        """Reject questions containing only whitespace."""
        if not value.strip():
            raise ValueError("Question must not be empty.")
        return value.strip()


class SourceChunk(BaseModel):
    """Retrieved context chunk returned with an answer."""

    source: str
    chunk_index: int
    timestamp: str
    text: str
    distance: float | None = None


class QueryResponse(BaseModel):
    """Generated answer and the context used to ground it."""

    answer: str
    sources: list[SourceChunk]
