from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk
from src.retrieval.tokenizer import SimpleTokenizer


class BM25Index:
    """
    In-memory BM25 index.

    Responsible for:

    - building the lexical index
    - storing chunk references
    - searching documents
    """

    def __init__(
        self,
        tokenizer: SimpleTokenizer | None = None,
    ) -> None:

        self.tokenizer = tokenizer or SimpleTokenizer()

        self._bm25: BM25Okapi | None = None

        self._chunks: list[Chunk] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(
        self,
        chunks: list[Chunk],
    ) -> None:
        """
        Build the BM25 index.
        """

        if not chunks:
            raise ValueError(
                "Cannot build BM25 index with no chunks."
            )

        self._chunks = chunks

        corpus = [

            self.tokenizer.tokenize(
                chunk.text,
            )

            for chunk in chunks

        ]

        self._bm25 = BM25Okapi(
            corpus,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """
        Search the BM25 index.

        Returns
        -------
        list[(Chunk, score)]
        """

        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        tokens = self.tokenizer.tokenize(
            query,
        )

        scores = self._bm25.get_scores(
            tokens,
        )

        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[:top_k]