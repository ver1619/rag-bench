from src.chunkers.base import BaseChunker
from src.models.chunk import Chunk
from src.models.document import Document


class FixedChunker(BaseChunker):
    """
    Splits a document into fixed-size character chunks.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:

        self._validate(chunk_size, chunk_overlap)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def name(self) -> str:
        return "Fixed"

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into fixed-size overlapping chunks.
        """

        if not document.text.strip():
            return []

        chunks: list[Chunk] = []

        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(document.text):

            end = min(start + self.chunk_size, len(document.text))

            chunk_text = document.text[start:end]

            # Skip whitespace-only chunks
            if chunk_text.strip():

                chunks.append(
                    Chunk(
                        chunk_id="",                  # Assigned later
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        start_offset=start,
                        end_offset=end,
                        text=chunk_text,
                        length=len(chunk_text),
                        title=document.title,
                        source=document.filename,
                        page=None,
                    )
                )

                chunk_index += 1

            start += step

        return chunks

    @staticmethod
    def _validate(
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )