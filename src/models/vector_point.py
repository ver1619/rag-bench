from typing import Any

from pydantic import BaseModel, Field


class VectorPoint(BaseModel):
    """
    Represents a point stored in a vector database.
    """

    id: int = Field(
        ...,
        description="Qdrant point identifier"
    )

    vector: list[float] = Field(
        ...,
        description="Dense vector"
    )

    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata"
    )