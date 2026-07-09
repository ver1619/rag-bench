from src.embeddings.base import BaseEmbedder
from src.models.chunk import Chunk
from src.models.embedding import Embedding
from src.validators.base import BaseValidator


class EmbeddingValidator(BaseValidator):
    """
    Validates generated embeddings.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: list[Embedding],
        embedder: BaseEmbedder,
    ) -> None:

        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder

    @property
    def name(self) -> str:
        return "Embedding"

    def validate(self) -> tuple[bool, list[str]]:

        messages: list[str] = []
        passed = True

        if not self.embeddings:
            return False, ["No embeddings generated."]

        messages.append(
            f"Embeddings found: {len(self.embeddings)}"
        )

        # ----------------------------------------------------
        # One embedding per chunk
        # ----------------------------------------------------

        if len(self.embeddings) != len(self.chunks):
            passed = False
            messages.append(
                "Number of embeddings does not match number of chunks."
            )
        else:
            messages.append(
                "Embedding count matches chunk count."
            )

        # ----------------------------------------------------
        # Chunk ID Validation
        # ----------------------------------------------------

        chunk_ids = {
            chunk.chunk_id
            for chunk in self.chunks
        }

        invalid_chunk_ids = [
            embedding.chunk_id
            for embedding in self.embeddings
            if embedding.chunk_id not in chunk_ids
        ]

        if invalid_chunk_ids:
            passed = False
            messages.append(
                f"Invalid chunk references: {len(invalid_chunk_ids)}"
            )
        else:
            messages.append(
                "Chunk references are valid."
            )

        # ----------------------------------------------------
        # Vector Dimension Validation
        # ----------------------------------------------------

        invalid_dimensions = [
            embedding.chunk_id
            for embedding in self.embeddings
            if len(embedding.vector) != self.embedder.dimension
        ]

        if invalid_dimensions:
            passed = False
            messages.append(
                f"Invalid embedding dimensions: {len(invalid_dimensions)}"
            )
        else:
            messages.append(
                "Embedding dimensions are valid."
            )

        # ----------------------------------------------------
        # Empty Vector Validation
        # ----------------------------------------------------

        empty_vectors = [
            embedding.chunk_id
            for embedding in self.embeddings
            if len(embedding.vector) == 0
        ]

        if empty_vectors:
            passed = False
            messages.append(
                f"Empty vectors: {len(empty_vectors)}"
            )
        else:
            messages.append(
                "No empty vectors found."
            )

        # ----------------------------------------------------
        # Model Name Validation
        # ----------------------------------------------------

        invalid_models = [
            embedding.chunk_id
            for embedding in self.embeddings
            if embedding.model_name != self.embedder.name
        ]

        if invalid_models:
            passed = False
            messages.append(
                f"Invalid model names: {len(invalid_models)}"
            )
        else:
            messages.append(
                "Model names are valid."
            )

        return passed, messages