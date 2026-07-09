from sentence_transformers import SentenceTransformer

from src.embeddings.base import BaseEmbedder
from src.models.chunk import Chunk
from src.models.embedding import Embedding


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    SentenceTransformer-based embedder.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        """
        Return embedding dimension.
        """
        return self.model.get_embedding_dimension()

    def embed(
        self,
        chunks: list[Chunk],
    ) -> list[Embedding]:

        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        vectors = self.model.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        dimension = self.dimension

        embeddings: list[Embedding] = []

        for chunk, vector in zip(chunks, vectors):

            embeddings.append(

                Embedding(
                    chunk=chunk,
                    vector=vector.tolist(),
                    dimension=dimension,
                    model_name=self.model_name,
                )
            )

        return embeddings


    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a search query.
        """

        vector = self.model.encode(
            query,
            convert_to_numpy=True,
        )

        return vector.tolist()