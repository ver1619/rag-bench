from math import ceil

from src.models.embedding import Embedding
from src.models.vector_point import VectorPoint
from src.vectordb.base import BaseVectorStore


class IndexingService:
    """
    Handles indexing of embeddings into a vector database.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        collection_name: str,
    ) -> None:

        self.vector_store = vector_store
        self.collection_name = collection_name

    def index(
        self,
        embeddings: list[Embedding],
        overwrite: bool = True,
        batch_size: int = 256,
    ) -> None:
        """
        Create a collection and index embeddings.

        Parameters
        ----------
        embeddings
            Embeddings to index.

        overwrite
            Recreate the collection before indexing.

        batch_size
            Number of vectors uploaded per request.
        """

        if not embeddings:
            raise ValueError("No embeddings to index.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        dimension = embeddings[0].dimension

        # -------------------------------------------------
        # Collection Management
        # -------------------------------------------------

        if overwrite:

            if self.vector_store.collection_exists(
                self.collection_name,
            ):
                self.vector_store.delete_collection(
                    self.collection_name,
                )

            self.vector_store.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
            )

        else:

            if not self.vector_store.collection_exists(
                self.collection_name,
            ):
                self.vector_store.create_collection(
                    collection_name=self.collection_name,
                    dimension=dimension,
                )

        # -------------------------------------------------
        # Batch Upload
        # -------------------------------------------------

        total_batches = ceil(len(embeddings) / batch_size)

        print(
            f"\nUploading {len(embeddings)} embeddings "
            f"in {total_batches} batches...\n"
        )

        for batch_number, start in enumerate(
            range(0, len(embeddings), batch_size),
            start=1,
        ):

            end = start + batch_size

            batch = embeddings[start:end]

            points = self._to_points(
                embeddings=batch,
                start_id=start + 1,
            )

            self.vector_store.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            print(
                f"[{batch_number:>3}/{total_batches}] "
                f"Indexed {len(batch)} vectors"
            )

        print("\nIndexing completed successfully.")

    def _to_points(
        self,
        embeddings: list[Embedding],
        start_id: int,
    ) -> list[VectorPoint]:
        """
        Convert embeddings into vector points.
        """

        points: list[VectorPoint] = []

        for offset, embedding in enumerate(embeddings):

            points.append(

                VectorPoint(
                    id=start_id + offset,
                    vector=embedding.vector,
                    payload=self._build_payload(
                        embedding,
                    ),
                )

            )

        return points

    def _build_payload(
        self,
        embedding: Embedding,
    ) -> dict:
        """
        Build payload stored alongside each vector.
        """

        chunk = embedding.chunk

        return {

        "chunk_id": chunk.chunk_id,

        "text": chunk.text,

        "document_id": chunk.document_id,

        "title": chunk.title,

        "source": chunk.source,

        "page": chunk.page,

        "chunk_index": chunk.chunk_index,

        "model_name": embedding.model_name,

        "dimension": embedding.dimension,

        }