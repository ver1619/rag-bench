from src.retrieval.base import BaseRetriever
from src.retrieval.models import RetrievalResponse

from src.rerankers.base import BaseReranker
from src.rerankers.models import RerankRequest


class RetrievalPipeline:
    """
    Retrieval pipeline.

    Executes a retriever followed by an optional reranker.

    Workflow
    --------
        Query
          │
          ▼
      Retriever
          │
          ▼
    RetrievalResponse
          │
          ▼
    Optional Reranker
          │
          ▼
    RetrievalResponse
    """

    def __init__(
        self,
        *,
        retriever: BaseRetriever,
        reranker: BaseReranker | None = None,
    ) -> None:

        self._retriever = retriever
        self._reranker = reranker

    @property
    def retriever(self) -> BaseRetriever:
        """
        Underlying retriever.
        """

        return self._retriever

    @property
    def reranker(self) -> BaseReranker | None:
        """
        Configured reranker.
        """

        return self._reranker

    @property
    def name(self) -> str:
        """
        Human-readable pipeline name.
        """

        if self._reranker is None:

            return self._retriever.name

        return (
            f"{self._retriever.name}"
            f" + "
            f"{self._reranker.name}"
        )

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
    ) -> RetrievalResponse:
        """
        Execute retrieval.

        Parameters
        ----------
        query
            User query.

        top_k
            Number of results requested.

        Returns
        -------
        RetrievalResponse
        """

        #
        # Retrieve
        #

        response = self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        #
        # No reranker
        #

        if self._reranker is None:

            return response

        #
        # Rerank
        #

        request = RerankRequest(
            query=query,
            response=response,
            top_k=top_k,
        )

        response = self._reranker.rerank(
            request=request,
        )

        return response