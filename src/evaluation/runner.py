from src.evaluation.aggregator import MetricsAggregator
from src.evaluation.dataset import EvaluationDataset
from src.evaluation.document_mapper import DocumentMapper
from src.evaluation.latency import Latency
from src.evaluation.metrics.base import BaseMetric
from src.evaluation.models import (
    BenchmarkResult,
    EvaluationResult,
    MetricResult,
)

from src.retrieval.pipeline import RetrievalPipeline


class EvaluationRunner:
    """
    Executes a retrieval benchmark over an evaluation dataset.
    """

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline,
        dataset: EvaluationDataset,
        metrics: list[BaseMetric],
        latency: Latency,
        mapper: DocumentMapper | None = None,
    ) -> None:

        self.retrieval_pipeline = retrieval_pipeline
        self.dataset = dataset
        self.metrics = metrics
        self.latency = latency
        self.mapper = mapper or DocumentMapper()

    def run(
        self,
        *,
        top_k: int = 5,
        limit: int | None = None,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """
        Execute the benchmark.

        Parameters
        ----------
        top_k
            Number of retrieved results.

        limit
            Maximum number of benchmark queries.

        show_progress
            Display benchmark progress.

        Returns
        -------
        BenchmarkResult
        """

        queries = self.dataset.load(
            limit=limit,
        )

        aggregator = MetricsAggregator()

        results: list[EvaluationResult] = []

        total_queries = len(queries)

        for index, query in enumerate(
            queries,
            start=1,
        ):

            if show_progress:

                print(
                    f"\rEvaluating [{index}/{total_queries}]",
                    end="",
                    flush=True,
                )

            response, latency_ms = self.latency.measure(
                retrieval_pipeline=self.retrieval_pipeline,
                query=query.query,
                top_k=top_k,
            )

            expected_documents = [
                document.document_id
                for document in query.relevant_documents
            ]

            retrieved_documents = self.mapper.map(
                response,
            )

            metric_results: list[MetricResult] = []

            metric_values: dict[str, float] = {}

            for metric in self.metrics:

                score = metric.compute(
                    expected=expected_documents,
                    retrieved=retrieved_documents,
                )

                metric_values[metric.name] = score

                metric_results.append(
                    MetricResult(
                        name=metric.name,
                        value=score,
                    )
                )

            metric_results.append(
                MetricResult(
                    name=self.latency.name,
                    value=latency_ms,
                )
            )

            aggregator.add(
                recall=metric_values.get(
                    "Recall@K",
                    0.0,
                ),
                precision=metric_values.get(
                    "Precision@K",
                    0.0,
                ),
                mrr=metric_values.get(
                    "MRR",
                    0.0,
                ),
                ndcg=metric_values.get(
                    "nDCG@K",
                    0.0,
                ),
                latency_ms=latency_ms,
            )

            results.append(
                EvaluationResult(
                    query=query,
                    retriever=self.retrieval_pipeline.name,
                    metrics=metric_results,
                )
            )

        if show_progress:

            print()

        return BenchmarkResult(
            retriever=self.retrieval_pipeline.name,
            total_queries=total_queries,
            results=results,
            summary=aggregator.summary(),
        )