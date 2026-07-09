import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt_tab is downloaded
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from src.chunkers.base import BaseChunker
from src.models.chunk import Chunk
from src.models.document import Document


class SentenceChunker(BaseChunker):
    """
    Groups sentences into chunks up to a maximum character length.
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
    ) -> None:

        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0.")

        self.max_chunk_size = max_chunk_size

    @property
    def name(self) -> str:
        return "Sentence"

    def chunk(self, document: Document) -> list[Chunk]:

        if not document.text.strip():
            return []

        sentences = sent_tokenize(document.text)

        chunks = []

        current = ""
        current_start = 0
        chunk_index = 0
        search_start = 0

        for sentence in sentences:

            if not current:
                pos = document.text.find(sentence, search_start)
                current_start = pos if pos != -1 else search_start

            candidate = sentence if not current else current + " " + sentence

            if len(candidate) <= self.max_chunk_size:

                current = candidate

            else:

                end = current_start + len(current)

                chunks.append(
                    Chunk(
                        chunk_id="",
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        start_offset=current_start,
                        end_offset=end,
                        text=current,
                        length=len(current),
                        title=document.title,
                        source=document.filename,
                        page=None,
                    )
                )

                chunk_index += 1

                pos = document.text.find(sentence, end)

                current_start = pos if pos != -1 else end
                current = sentence

            search_start = current_start

        if current:

            end = current_start + len(current)

            chunks.append(
                Chunk(
                    chunk_id="",
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    start_offset=current_start,
                    end_offset=end,
                    text=current,
                    length=len(current),
                    title=document.title,
                    source=document.filename,
                    page=None,
                )
            )

        return chunks