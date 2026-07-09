from abc import ABC
from abc import abstractmethod

from src.retrieval.models import RetrievalResponse
from src.rerankers.models import RerankRequest


class BaseReranker(ABC):
    """
    Base interface for all rerankers.

    A reranker receives the initial retrieval results and
    returns a reordered subset of those results based on
    semantic relevance.

    Rerankers do not retrieve new documents—they only
    reorder existing retrieval results.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable reranker name.
        """
        ...

    @abstractmethod
    def rerank(
        self,
        *,
        request: RerankRequest,
    ) -> RetrievalResponse:
        """
        Rerank retrieved results.

        Parameters
        ----------
        query
            Original user query.

        response
            Initial retrieval results.

        top_k
            Number of results to return after reranking.

        Returns
        -------
        RetrievalResponse
            Reranked retrieval results.
        """
        ...