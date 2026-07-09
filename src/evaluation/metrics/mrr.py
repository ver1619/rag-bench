from src.evaluation.metrics.base import BaseMetric


class MeanReciprocalRank(BaseMetric):
    """
    Computes Reciprocal Rank.
    """

    @property
    def name(self) -> str:
        return "MRR"

    def compute(
        self,
        *,
        expected: list[str],
        retrieved: list[str],
    ) -> float:

        relevant = set(expected)

        for rank, document in enumerate(
            retrieved,
            start=1,
        ):

            if document in relevant:
                return 1.0 / rank

        return 0.0