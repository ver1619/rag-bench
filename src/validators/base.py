from abc import ABC, abstractmethod


class BaseValidator(ABC):
    """
    Base interface for all validation stages.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the validator.
        """
        pass

    @abstractmethod
    def validate(self) -> tuple[bool, list[str]]:
        """
        Run validation.

        Returns
        -------
        tuple[bool, list[str]]

        bool
            Overall validation status.

        list[str]
            Validation messages.
        """
        pass