from src.models.chunk import Chunk
from src.models.document import Document
from src.validators.base import BaseValidator


class ChunkingValidator(BaseValidator):
    """
    Validates generated chunks.
    """

    def __init__(
        self,
        documents: list[Document],
        chunks: list[Chunk],
    ) -> None:

        self.documents = documents
        self.chunks = chunks

    @property
    def name(self) -> str:
        return "Chunking"

    def validate(self) -> tuple[bool, list[str]]:

        messages: list[str] = []
        passed = True

        if not self.chunks:
            return False, ["No chunks generated."]

        messages.append(f"Chunks found: {len(self.chunks)}")

        # ----------------------------------------------------
        # Unique Chunk IDs
        # ----------------------------------------------------

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            passed = False
            messages.append("Duplicate chunk IDs detected.")
        else:
            messages.append("Chunk IDs are unique.")

        # ----------------------------------------------------
        # Parent Document Validation
        # ----------------------------------------------------

        document_ids = {
            document.document_id
            for document in self.documents
        }

        invalid_documents = [
            chunk.chunk_id
            for chunk in self.chunks
            if chunk.document_id not in document_ids
        ]

        if invalid_documents:
            passed = False
            messages.append(
                f"Invalid parent documents: {len(invalid_documents)}"
            )
        else:
            messages.append("Parent document references are valid.")

        # ----------------------------------------------------
        # Offset Validation
        # ----------------------------------------------------

        invalid_offsets = [
            chunk.chunk_id
            for chunk in self.chunks
            if (
                chunk.start_offset < 0
                or chunk.end_offset <= chunk.start_offset
            )
        ]

        if invalid_offsets:
            passed = False
            messages.append(
                f"Invalid offsets: {len(invalid_offsets)}"
            )
        else:
            messages.append("Offsets are valid.")

        # ----------------------------------------------------
        # Length Validation
        # ----------------------------------------------------

        invalid_lengths = [
            chunk.chunk_id
            for chunk in self.chunks
            if chunk.length != len(chunk.text)
        ]

        if invalid_lengths:
            passed = False
            messages.append(
                f"Incorrect chunk lengths: {len(invalid_lengths)}"
            )
        else:
            messages.append("Chunk lengths are valid.")

        # ----------------------------------------------------
        # Empty Chunks
        # ----------------------------------------------------

        empty_chunks = [
            chunk.chunk_id
            for chunk in self.chunks
            if not chunk.text.strip()
        ]

        if empty_chunks:
            passed = False
            messages.append(
                f"Empty chunks: {len(empty_chunks)}"
            )
        else:
            messages.append("No empty chunks found.")

        # ----------------------------------------------------
        # Chunk Index Validation
        # ----------------------------------------------------

        invalid_indices = [
            chunk.chunk_id
            for chunk in self.chunks
            if chunk.chunk_index < 0
        ]

        if invalid_indices:
            passed = False
            messages.append(
                f"Invalid chunk indices: {len(invalid_indices)}"
            )
        else:
            messages.append("Chunk indices are valid.")

        return passed, messages