from pydantic import BaseModel, Field

from src.models.chunk import Chunk


class Embedding(BaseModel):
    """
    Dense vector embedding generated from a chunk.
    """

    chunk: Chunk = Field(
        ...,
        description="Source chunk.",
    )

    vector: list[float] = Field(
        ...,
        description="Embedding vector.",
    )

    dimension: int = Field(
        ...,
        ge=1,
        description="Embedding dimension.",
    )

    model_name: str = Field(
        ...,
        description="Embedding model.",
    )

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id