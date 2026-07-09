import time

from src.retrieval.base import BaseRetriever
from src.retrieval.models import RetrievalResponse


class Latency:
    """
    Measures retrieval latency.

    Unlike the other evaluation metrics, latency must
    execute the retriever in order to measure elapsed
    execution time.
    """

    @property
    def name(self) -> str:
        return "Latency (ms)"

    def measure(
        self,
        *,
        retrieval_pipeline: BaseRetriever,
        query: str,
        top_k: int,
    ) -> tuple[RetrievalResponse, float]:
        """
        Execute retrieval and measure latency.

        Returns
        -------
        (RetrievalResponse, latency_ms)
        """

        start = time.perf_counter()

        response = retrieval_pipeline.retrieve(
            query=query,
            top_k=top_k,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        return response, elapsed_ms