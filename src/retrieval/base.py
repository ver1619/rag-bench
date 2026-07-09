from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResponse


class BaseRetriever(ABC):
    """
    Base interface for all retrieval strategies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the retrieval strategy.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResponse:
        """
        Retrieve the top-k most relevant results.
        """
        pass