from sentence_transformers import CrossEncoder

from src.rerankers.base import BaseReranker
from src.rerankers.models import (
    RerankerConfig,
    RerankRequest,
)

from src.retrieval.models import (
    RetrievalResponse,
    RetrievalResult,
)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based reranker.

    A CrossEncoder jointly encodes the query and each
    retrieved chunk to estimate semantic relevance.

    Unlike dense retrieval, no embeddings are produced.
    The model simply assigns a relevance score to each
    retrieved result and returns a reranked response.
    """

    def __init__(
        self,
        config: RerankerConfig,
    ) -> None:

        self._config = config

        self._model: CrossEncoder | None = None

    @property
    def name(self) -> str:

        model_name = self._config.model_name.lower()
        if "bge" in model_name:
            return "BGE Reranker"
        elif "mxbai" in model_name:
            return "MXBAI Reranker"
        return "CrossEncoder"

    @property
    def model_name(self) -> str:

        return self._config.model_name

    @property
    def model(self) -> CrossEncoder:
        """
        Lazily load the CrossEncoder model.
        """

        if self._model is None:

            self._model = CrossEncoder(
                self._config.model_name,
            )

        return self._model

    def rerank(
        self,
        request: RerankRequest,
    ) -> RetrievalResponse:
        """
        Rerank retrieval results.
        """

        results = request.response.results

        if not results:

            return request.response

        top_k = min(
            request.top_k,
            len(results),
        )

        sentence_pairs = self._build_pairs(
            request.query,
            results,
        )

        scores = self.model.predict(
            sentence_pairs,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
        )

        reranked = self._rerank(
            results,
            scores,
            top_k,
        )

        return RetrievalResponse(
            query=request.query,
            retriever=f"{request.response.retriever}+CrossEncoder",
            results=reranked,
        )

    def _build_pairs(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[tuple[str, str]]:
        """
        Construct (query, chunk) pairs for inference.
        """

        return [

            (
                query,
                result.payload.text,
            )

            for result in results

        ]

    def _rerank(
        self,
        results: list[RetrievalResult],
        scores,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Sort retrieval results by CrossEncoder score.
        """

        ranked = sorted(
            zip(results, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        reranked: list[RetrievalResult] = []

        for rank, (result, score) in enumerate(
            ranked[:top_k],
            start=1,
        ):

            reranked.append(

                RetrievalResult(

                    chunk_id=result.chunk_id,

                    score=float(score),

                    rank=rank,

                    payload=result.payload,

                )

            )

        return reranked