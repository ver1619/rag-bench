import pandas as pd

from src.evaluation.models import BenchmarkResult


class BenchmarkAnalysis:
    """
    Utility functions for analysing benchmark results.

    This class converts BenchmarkResult objects into a
    pandas DataFrame that can be used directly inside
    Jupyter notebooks.
    """

    @staticmethod
    def to_dataframe(
        benchmark: BenchmarkResult,
    ) -> pd.DataFrame:
        """
        Convert benchmark results into a DataFrame.

        Parameters
        ----------
        benchmark
            Benchmark execution result.

        Returns
        -------
        pandas.DataFrame
        """

        rows: list[dict] = []

        for evaluation in benchmark.results:

            row = {

                "query_id": evaluation.query.query_id,

                "query": evaluation.query.query,

                "pipeline": benchmark.retriever,

            }

            for metric in evaluation.metrics:

                row[metric.name] = metric.value

            rows.append(row)

        return pd.DataFrame(rows)