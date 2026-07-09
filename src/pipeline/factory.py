from src.chunkers.base import BaseChunker
from src.chunkers.fixed import FixedChunker
from src.chunkers.recursive import RecursiveChunker
from src.chunkers.sentence import SentenceChunker

from src.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)

from src.pipeline.builder import PipelineBuilder


CHUNKER_REGISTRY = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
}


def _create_chunker(
    chunker: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> BaseChunker:
    """
    Instantiate a chunker by name.

    Parameters
    ----------
    chunker
        One of "fixed", "recursive", "sentence".

    chunk_size
        Maximum chunk size in characters.

    chunk_overlap
        Overlap between consecutive chunks (not used by SentenceChunker).
    """

    if chunker not in CHUNKER_REGISTRY:

        raise ValueError(
            f"Unknown chunker '{chunker}'. "
            f"Available: {list(CHUNKER_REGISTRY.keys())}"
        )

    if chunker == "sentence":
        return SentenceChunker(
            max_chunk_size=chunk_size,
        )

    return CHUNKER_REGISTRY[chunker](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def create_pipeline_builder(
    model_name: str,
    chunker: str = "fixed",
) -> PipelineBuilder:
    """
    Create the document-processing pipeline.

    Parameters
    ----------
    model_name
        HuggingFace embedding model name.

    chunker
        Chunking strategy: "fixed", "recursive", or "sentence".
    """

    return PipelineBuilder(

        chunker=_create_chunker(chunker),

        embedder=SentenceTransformerEmbedder(
            model_name=model_name,
        ),

    )