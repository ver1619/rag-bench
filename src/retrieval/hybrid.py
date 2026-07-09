from src.retrieval.base import BaseRetriever
from src.retrieval.fusion.base import BaseFusion
from src.retrieval.models import RetrievalResponse


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines multiple retrieval
    strategies using a fusion algorithm.
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        fusion: BaseFusion,
    ) -> None:

        if not retrievers:
            raise ValueError(
                "HybridRetriever requires at least one retriever."
            )

        self.retrievers = retrievers
        self.fusion = fusion

    @property
    def name(self) -> str:
        return f"Hybrid ({self.fusion.name})"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResponse:
        """
        Retrieve using every retriever and fuse
        the rankings.
        """

        responses = [

            retriever.retrieve(
                query=query,
                top_k=top_k,
            )

            for retriever in self.retrievers

        ]

        return self.fusion.fuse(
            responses=responses,
            top_k=top_k,
        )