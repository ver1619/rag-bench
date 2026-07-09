from src.evaluation.metrics.base import BaseMetric


class RecallAtK(BaseMetric):
    """
    Computes Recall@K.

    Recall measures the fraction of relevant documents
    retrieved within the top-k results.
    """

    @property
    def name(self) -> str:
        return "Recall@K"

    def compute(
        self,
        *,
        expected: list[str],
        retrieved: list[str],
    ) -> float:

        if not expected:
            return 0.0

        relevant = set(expected)

        retrieved_set = set(retrieved)

        retrieved_relevant = len(
            relevant.intersection(retrieved_set)
        )

        return retrieved_relevant / len(relevant)