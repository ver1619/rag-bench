from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunkers.base import BaseChunker
from src.models.chunk import Chunk
from src.models.document import Document


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based chunker using LangChain.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:

        self._validate(chunk_size, chunk_overlap)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = [
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    @property
    def name(self) -> str:
        return "Recursive"

    def chunk(self, document: Document) -> list[Chunk]:

        if not document.text.strip():
            return []

        texts = self.splitter.split_text(document.text)

        chunks = []
        search_start = 0

        for index, text in enumerate(texts):

            start = document.text.find(text, search_start)

            if start == -1:
                start = search_start

            end = start + len(text)

            chunks.append(
                Chunk(
                    chunk_id="",
                    document_id=document.document_id,
                    chunk_index=index,
                    start_offset=start,
                    end_offset=end,
                    text=text,
                    length=len(text),
                    title=document.title,
                    source=document.filename,
                    page=None,
                )
            )

            search_start = end

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