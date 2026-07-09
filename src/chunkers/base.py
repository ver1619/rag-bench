from abc import ABC, abstractmethod

from src.models.document import Document
from src.models.chunk import Chunk


class BaseChunker(ABC):
    """
    Base interface for all chunking strategies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the chunking strategy.
        """
        pass

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.

        Parameters
        ----------
        document : Document
            Input document.

        Returns
        -------
        list[Chunk]
            Generated chunks.
        """
        pass