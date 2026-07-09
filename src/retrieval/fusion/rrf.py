from collections import defaultdict

from src.retrieval.fusion.base import BaseFusion
from src.retrieval.models import (
    RetrievalResponse,
    RetrievalResult,
)


class RRFFusion(BaseFusion):
    """
    Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        k: int = 60,
    ) -> None:

        self.k = k

    @property
    def name(self) -> str:
        return "Reciprocal Rank Fusion"

    def fuse(
        self,
        responses: list[RetrievalResponse],
        top_k: int,
    ) -> RetrievalResponse:

        scores = defaultdict(float)
        payloads = {}

        for response in responses:

            for result in response.results:

                scores[result.chunk_id] += (
                    1.0 /
                    (self.k + result.rank)
                )
                if result.chunk_id not in payloads:
                    payloads[result.chunk_id] = result.payload

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for rank, (chunk_id, score) in enumerate(
            ranked[:top_k],
            start=1,
        ):

            results.append(

                RetrievalResult(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=score,
                    payload=payloads[chunk_id],
                )

            )

        return RetrievalResponse(
            query=responses[0].query,
            retriever=self.name,
            results=results,
        )