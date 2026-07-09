from src.retrieval.base import BaseRetriever
from src.retrieval.index import BM25Index
from src.retrieval.models import (
    RetrievalPayload,
    RetrievalResponse,
    RetrievalResult,
)


class BM25Retriever(BaseRetriever):
    """
    BM25 lexical retriever.
    """

    def __init__(
        self,
        index: BM25Index,
    ) -> None:

        self.index = index

    @property
    def name(self) -> str:
        return "BM25 Retrieval"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResponse:

        search_results = self.index.search(
            query=query,
            top_k=top_k,
        )

        results: list[RetrievalResult] = []

        for rank, (chunk, score) in enumerate(
            search_results,
            start=1,
        ):

            payload = RetrievalPayload(
                text=chunk.text,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                metadata={},
            )

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    rank=rank,
                    payload=payload,
                )
            )

        return RetrievalResponse(
            query=query,
            retriever=self.name,
            results=results,
        )