from src.validators.base import BaseValidator


class PipelineValidator:
    """
    Orchestrates all validation stages.
    """

    def __init__(
        self,
        validators: list[BaseValidator],
    ) -> None:

        self.validators = validators

    def validate(self) -> tuple[bool, dict[str, list[str]]]:
        """
        Execute all validators.

        Returns
        -------
        tuple[bool, dict[str, list[str]]]

        bool
            Overall validation status.

        dict
            Validation report grouped by validator.
        """

        overall_status = True
        report: dict[str, list[str]] = {}

        for validator in self.validators:

            passed, messages = validator.validate()

            report[validator.name] = messages

            if not passed:
                overall_status = False

        return overall_status, report