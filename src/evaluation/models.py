from pydantic import BaseModel, Field

from src.evaluation.aggregator import EvaluationSummary


class RelevantDocument(BaseModel):
    """
    Represents one relevant document for a benchmark query.
    """

    document_id: str = Field(
        ...,
        description="Relevant document identifier.",
    )


class EvaluationQuery(BaseModel):
    """
    One benchmark query together with its ground-truth documents.
    """

    query_id: str = Field(
        ...,
        description="Unique benchmark query identifier.",
    )

    query: str = Field(
        ...,
        description="Benchmark query.",
    )

    relevant_documents: list[RelevantDocument] = Field(
        default_factory=list,
        description="Ground-truth relevant documents.",
    )


class MetricResult(BaseModel):
    """
    Result produced by one evaluation metric.
    """

    name: str = Field(
        ...,
        description="Metric name.",
    )

    value: float = Field(
        ...,
        description="Metric value.",
    )


class EvaluationResult(BaseModel):
    """
    Evaluation result for one benchmark query.
    """

    query: EvaluationQuery = Field(
        ...,
        description="Benchmark query.",
    )

    retriever: str = Field(
        ...,
        description="Retriever name.",
    )

    metrics: list[MetricResult] = Field(
        default_factory=list,
        description="Computed metrics.",
    )


class BenchmarkResult(BaseModel):
    """
    Complete benchmark result.
    """

    retriever: str = Field(
        ...,
        description="Retriever evaluated.",
    )

    total_queries: int = Field(
        ...,
        ge=0,
        description="Number of benchmark queries.",
    )

    results: list[EvaluationResult] = Field(
        default_factory=list,
        description="Per-query evaluation results.",
    )

    summary: EvaluationSummary = Field(
        ...,
        description="Aggregated benchmark metrics.",
    )