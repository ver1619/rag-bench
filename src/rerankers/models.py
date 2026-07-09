from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.retrieval.models import RetrievalResponse


class RerankerConfig(BaseModel):
    """
    Configuration shared by all rerankers.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    model_name: str = Field(
        ...,
        description="Underlying reranker model.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        description="Default number of reranked results.",
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        description="Inference batch size.",
    )


class RerankRequest(BaseModel):
    """
    Input passed to a reranker.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    query: str = Field(
        ...,
        description="Original search query.",
    )

    response: RetrievalResponse = Field(
        ...,
        description="Initial retrieval results.",
    )

    top_k: int = Field(
        ...,
        ge=1,
        description="Number of reranked results to return.",
    )