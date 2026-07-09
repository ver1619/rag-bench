from typing import Any

from pydantic import BaseModel, Field


class RetrievalPayload(BaseModel):
    """
    Structured payload associated with a retrieved chunk.
    """

    text: str = Field(
        ...,
        description="Retrieved chunk text."
    )

    document_id: str | None = Field(
        default=None,
        description="Parent document identifier."
    )

    title: str | None = Field(
        default=None,
        description="Document title."
    )

    source: str | None = Field(
        default=None,
        description="Document source or file path."
    )

    page: int | None = Field(
        default=None,
        description="Page number within the document."
    )

    chunk_index: int | None = Field(
        default=None,
        description="Chunk index within the parent document."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional retriever-specific metadata."
    )


class RetrievalResult(BaseModel):
    """
    Represents a single retrieved result.
    """

    chunk_id: str = Field(
        ...,
        description="Retrieved chunk identifier."
    )

    score: float = Field(
        ...,
        description="Retrieval score."
    )

    rank: int = Field(
        ...,
        description="Rank of the retrieved result."
    )

    payload: RetrievalPayload = Field(
        ...,
        description="Structured payload returned by the retriever."
    )


class RetrievalResponse(BaseModel):
    """
    Complete response produced by a retriever.
    """

    query: str = Field(
        ...,
        description="Original user query."
    )

    retriever: str = Field(
        ...,
        description="Retriever that produced the response."
    )

    results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Retrieved results."
    )