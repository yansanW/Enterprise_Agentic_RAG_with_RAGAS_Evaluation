# src/pipeline/schemas.py
from pydantic import BaseModel, Field


class GuardedAnswerSchema(BaseModel):
    answer: str = Field(
        description="The verified factual answer extracted STRICTLY from the provided document chunks. "
        "If the context does not contain sufficient evidence to answer, you MUST return exactly: "
        "'Information not found within verified knowledge base.'"
    )
    is_supported_by_context: bool = Field(
        description="Set to True if the answer is completely backed by the retrieved text chunks. "
        "Set to False if the context lacks sufficient evidence or requires extrapolation."
    )
    citations: list[str] = Field(
        description="An array of exact, verbatim text snippets or source metadata tags used directly to build this answer. "
        "Leave empty if information is not found."
    )
