import json
from pathlib import Path

from pydantic import ValidationError

from src.evaluation.models import EvaluationQuery


class EvaluationDataset:
    """
    Loads benchmark queries from disk.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:

        self.path = Path(path)

        self._queries: list[EvaluationQuery] | None = None

    def load(
        self,
        limit: int | None = None,
    ) -> list[EvaluationQuery]:
        """
        Load benchmark queries.
        """

        if self._queries is None:

            self._queries = self._read()

        if limit is None:

            return self._queries

        return self._queries[:limit]

    def reload(self) -> None:
        """
        Force reload from disk.
        """

        self._queries = None

    def _read(
        self,
    ) -> list[EvaluationQuery]:

        if not self.path.exists():

            raise FileNotFoundError(
                f"Evaluation dataset not found: {self.path}"
            )

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if not isinstance(
            data,
            list,
        ):
            raise ValueError(
                "Evaluation dataset must contain a JSON array."
            )

        queries: list[EvaluationQuery] = []

        for index, item in enumerate(
            data,
            start=1,
        ):

            try:

                queries.append(
                    EvaluationQuery(**item)
                )

            except ValidationError as exc:

                raise ValueError(
                    f"Invalid evaluation query at index {index}."
                ) from exc

        return queries

    def __len__(self) -> int:

        return len(self.load())