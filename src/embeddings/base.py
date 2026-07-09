from abc import ABC, abstractmethod

from src.models.chunk import Chunk
from src.models.embedding import Embedding


class BaseEmbedder(ABC):
    """
    Base interface for all embedding models.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the embedding model.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Output embedding dimension.
        """
        pass

    @abstractmethod
    def embed(
        self,
        chunks: list[Chunk],
    ) -> list[Embedding]:
        """
        Generate embeddings for a batch of chunks.
        """
        pass

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a search query.
        """
        pass