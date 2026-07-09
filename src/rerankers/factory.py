from src.rerankers.base import BaseReranker
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.rerankers.models import RerankerConfig


DEFAULT_CROSS_ENCODER = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

DEFAULT_BGE_RERANKER = (
    "BAAI/bge-reranker-base"
)

DEFAULT_MXBAI_RERANKER = (
    "mixedbread-ai/mxbai-rerank-base-v1"
)


def create_cross_encoder_reranker(
    *,
    model_name: str = DEFAULT_CROSS_ENCODER,
    top_k: int = 5,
    batch_size: int = 32,
) -> BaseReranker:
    """
    Create a CrossEncoder reranker (ms-marco default).

    Parameters
    ----------
    model_name
        HuggingFace CrossEncoder model.

    top_k
        Default reranked results.

    batch_size
        CrossEncoder inference batch size.

    Returns
    -------
    BaseReranker
    """

    config = RerankerConfig(

        model_name=model_name,

        top_k=top_k,

        batch_size=batch_size,

    )

    return CrossEncoderReranker(
        config=config,
    )


def create_bge_reranker(
    *,
    model_name: str = DEFAULT_BGE_RERANKER,
    top_k: int = 5,
    batch_size: int = 32,
) -> BaseReranker:
    """
    Create a BGE reranker (bge-reranker-base).
    """

    config = RerankerConfig(
        model_name=model_name,
        top_k=top_k,
        batch_size=batch_size,
    )

    return CrossEncoderReranker(
        config=config,
    )


def create_mxbai_reranker(
    *,
    model_name: str = DEFAULT_MXBAI_RERANKER,
    top_k: int = 5,
    batch_size: int = 32,
) -> BaseReranker:
    """
    Create an mxbai reranker (mxbai-rerank-base-v1).
    """

    config = RerankerConfig(
        model_name=model_name,
        top_k=top_k,
        batch_size=batch_size,
    )

    return CrossEncoderReranker(
        config=config,
    )