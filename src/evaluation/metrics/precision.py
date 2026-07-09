from src.evaluation.metrics.base import BaseMetric


class PrecisionAtK(BaseMetric):
    """
    Computes Precision@K.
    """

    @property
    def name(self) -> str:
        return "Precision@K"

    def compute(
        self,
        *,
        expected: list[str],
        retrieved: list[str],
    ) -> float:

        if not retrieved:
            return 0.0

        relevant = set(expected)

        retrieved_relevant = sum(
            document in relevant
            for document in retrieved
        )

        return retrieved_relevant / len(retrieved)