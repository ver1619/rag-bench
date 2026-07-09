from abc import ABC, abstractmethod

from src.retrieval.models import (
    RetrievalResponse,
)


class BaseFusion(ABC):
    """
    Base class for rank-fusion algorithms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the fusion strategy.
        """
        ...

    @abstractmethod
    def fuse(
        self,
        responses: list[RetrievalResponse],
        top_k: int,
    ) -> RetrievalResponse:
        """
        Fuse multiple retrieval responses into a single ranking.
        """
        ...