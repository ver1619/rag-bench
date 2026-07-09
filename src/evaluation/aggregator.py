from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """
    Aggregated benchmark metrics.
    """

    queries: int

    recall: float

    precision: float

    mrr: float

    ndcg: float

    latency_ms: float


class MetricsAggregator:
    """
    Aggregates evaluation metrics across benchmark queries.
    """

    def __init__(self) -> None:

        self._recall: list[float] = []
        self._precision: list[float] = []
        self._mrr: list[float] = []
        self._ndcg: list[float] = []
        self._latency: list[float] = []

    def add(
        self,
        *,
        recall: float,
        precision: float,
        mrr: float,
        ndcg: float,
        latency_ms: float,
    ) -> None:
        """
        Add metrics for one benchmark query.
        """

        self._recall.append(recall)
        self._precision.append(precision)
        self._mrr.append(mrr)
        self._ndcg.append(ndcg)
        self._latency.append(latency_ms)

    def summary(self) -> EvaluationSummary:
        """
        Compute average benchmark metrics.
        """

        queries = len(self._recall)

        if queries == 0:

            return EvaluationSummary(
                queries=0,
                recall=0.0,
                precision=0.0,
                mrr=0.0,
                ndcg=0.0,
                latency_ms=0.0,
            )

        return EvaluationSummary(
            queries=queries,
            recall=self._average(self._recall),
            precision=self._average(self._precision),
            mrr=self._average(self._mrr),
            ndcg=self._average(self._ndcg),
            latency_ms=self._average(self._latency),
        )

    @staticmethod
    def _average(
        values: list[float],
    ) -> float:

        return sum(values) / len(values)