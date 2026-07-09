from src.embeddings.base import BaseEmbedder
from src.retrieval.base import BaseRetriever
from src.retrieval.models import (
    RetrievalPayload,
    RetrievalResponse,
    RetrievalResult,
)
from src.vectordb.base import BaseVectorStore


class DenseRetriever(BaseRetriever):
    """
    Dense vector retriever backed by a vector database.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        collection_name: str,
    ) -> None:

        self.vector_store = vector_store
        self.embedder = embedder
        self.collection_name = collection_name

    @property
    def name(self) -> str:
        return "Dense Retrieval"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResponse:

        query_vector = self.embedder.embed_query(query)

        search_results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
        )

        results: list[RetrievalResult] = []

        for rank, point in enumerate(
            search_results,
            start=1,
        ):

            payload = point.payload or {}

            retrieval_payload = RetrievalPayload(
                text=payload.get("text", ""),
                document_id=payload.get("document_id"),
                title=payload.get("title"),
                source=payload.get("source"),
                page=payload.get("page"),
                chunk_index=payload.get("chunk_index"),
                metadata=payload.get("metadata", {}),
            )

            results.append(
                RetrievalResult(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    score=float(point.score),
                    rank=rank,
                    payload=retrieval_payload,
                )
            )

        return RetrievalResponse(
            query=query,
            retriever=self.name,
            results=results,
        )