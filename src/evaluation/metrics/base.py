from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for all evaluation metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable metric name.
        """
        ...

    @abstractmethod
    def compute(
        self,
        *,
        expected: list[str],
        retrieved: list[str],
    ) -> float:
        """
        Compute a metric for a single benchmark query.

        Parameters
        ----------
        expected
            Ground-truth document identifiers.

        retrieved
            Ordered retrieved document identifiers.

        Returns
        -------
        float
            Metric score.
        """
        ...