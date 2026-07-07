from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.models.vector_point import VectorPoint
from src.vectordb.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation of BaseVectorStore.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:

        from src.settings.qdrant import QdrantSettings
        settings = QdrantSettings()

        self.host = host if host is not None else settings.host
        self.port = port if port is not None else settings.port

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=60.0,
        )

    @property
    def name(self) -> str:
        return "Qdrant"

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    def health_check(
        self,
    ) -> bool:
        """
        Verify that the Qdrant server is reachable.
        """

        try:

            self.client.get_collections()

            return True

        except Exception:

            return False

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:
        """
        Create a new collection.
        """

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        Delete a collection if it exists.
        """

        if self.collection_exists(
            collection_name,
        ):

            self.client.delete_collection(
                collection_name=collection_name,
            )

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """
        Check whether a collection exists.
        """

        return self.client.collection_exists(
            collection_name=collection_name,
        )

    def upsert(
        self,
        collection_name: str,
        points: list[VectorPoint],
    ) -> None:
        """
        Insert or update vector points.
        """

        qdrant_points = [

            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )

            for point in points

        ]

        self.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
            wait=True,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
    ):
        """
        Search the collection using dense vector similarity.
        """

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return response.points