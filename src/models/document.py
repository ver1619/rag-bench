from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a parsed document after ingestion.

    This is the canonical object exchanged between all stages
    of the retrieval benchmark pipeline.
    """

    document_id: str = Field(..., description="Unique document identifier")

    filename: str = Field(
        ...,
        description="Original filename"
    )

    title: str = Field(
        ...,
        description="Document title"
    )

    pages: int = Field(
        ...,
        ge=1,
        description="Number of pages"
    )

    file_size: int = Field(
        ...,
        ge=0,
        description="File size in bytes"
    )

    text: str = Field(
        ...,
        description="Entire extracted document text"
    )