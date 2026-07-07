from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Represents a chunk generated from a document.
    """

    chunk_id: str = Field(
        default="",
        description="Unique chunk identifier (assigned by orchestrator)"
    )

    document_id: str = Field(
        ...,
        description="Parent document identifier"
    )

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of chunk within document"
    )

    start_offset: int = Field(
        ...,
        ge=0,
        description="Starting character offset"
    )

    end_offset: int = Field(
        ...,
        ge=0,
        description="Ending character offset"
    )

    text: str = Field(
        ...,
        description="Chunk content"
    )

    length: int = Field(
        ...,
        ge=0,
        description="Length of chunk"
    )

    title: str | None = Field(
        default=None,
        description="Document title"
    )

    source: str | None = Field(
        default=None,
        description="Source filename or URL"
    )

    page: int | None = Field(
        default=None,
        description="Page number if applicable"
    )