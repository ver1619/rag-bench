from src.embeddings.base import BaseEmbedder
from src.models.chunk import Chunk

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

from src.retrieval.index import BM25Index

from src.retrieval.tokenizer import SimpleTokenizer

from src.retrieval.fusion.rrf import RRFFusion

from src.vectordb.base import BaseVectorStore


def create_dense_retriever(
    *,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    collection_name: str,
) -> DenseRetriever:

    return DenseRetriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name=collection_name,
    )


def create_bm25_retriever(
    *,
    chunks: list[Chunk],
    tokenizer: SimpleTokenizer | None = None,
) -> BM25Retriever:

    index = BM25Index(
        tokenizer=tokenizer,
    )

    index.build(
        chunks,
    )

    return BM25Retriever(
        index=index,
    )


def create_hybrid_retriever(
    *,
    vector_store: BaseVectorStore,
    embedder: BaseEmbedder,
    collection_name: str,
    chunks: list[Chunk],
    tokenizer: SimpleTokenizer | None = None,
    rrf_k: int = 60,
) -> HybridRetriever:

    dense = create_dense_retriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name=collection_name,
    )

    bm25 = create_bm25_retriever(
        chunks=chunks,
        tokenizer=tokenizer,
    )

    fusion = RRFFusion(
        k=rrf_k,
    )

    return HybridRetriever(
        retrievers=[
            dense,
            bm25,
        ],
        fusion=fusion,
    )