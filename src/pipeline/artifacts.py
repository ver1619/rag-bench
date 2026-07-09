from pydantic import BaseModel

from src.models.document import Document
from src.models.chunk import Chunk
from src.models.embedding import Embedding


class PipelineArtifacts(BaseModel):
    """
    Stores all artifacts produced by the pipeline.
    """

    documents: list[Document]
    chunks: list[Chunk]
    embeddings: list[Embedding]