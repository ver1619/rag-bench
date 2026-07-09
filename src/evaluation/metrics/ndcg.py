from math import log2

from src.evaluation.metrics.base import BaseMetric


class NormalizedDiscountedCumulativeGain(BaseMetric):
    """
    Computes nDCG@K.
    """

    @property
    def name(self) -> str:
        return "nDCG@K"

    def compute(
        self,
        *,
        expected: list[str],
        retrieved: list[str],
    ) -> float:

        relevant = set(expected)

        dcg = 0.0

        for rank, document in enumerate(
            retrieved,
            start=1,
        ):

            if document in relevant:

                dcg += 1 / log2(rank + 1)

        ideal = min(
            len(expected),
            len(retrieved),
        )

        if ideal == 0:
            return 0.0

        idcg = sum(
            1 / log2(rank + 1)
            for rank in range(1, ideal + 1)
        )

        return dcg / idcg