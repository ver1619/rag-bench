from abc import ABC, abstractmethod
from src.models.vector_point import VectorPoint


class BaseVectorStore(ABC):
    """
    Base interface for vector database implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the vector database.
        """
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:
        """
        Create a vector collection.
        """
        pass

    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        Delete a collection.
        """
        pass

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        points: list[VectorPoint],
    ) -> None:
        """
        Insert or update embeddings.
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
    ):
        """
        Perform vector similarity search.
        """
        pass

    @abstractmethod
    def collection_exists(
            self,
            collection_name: str,
    ) -> bool:
        """
        Check whether a collection exists.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify that the vector database is reachable.
        """
        pass