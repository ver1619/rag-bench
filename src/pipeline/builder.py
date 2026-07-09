from src.ingestion.service import ingest_documents

from src.chunkers.base import BaseChunker
from src.embeddings.base import BaseEmbedder

from src.models.chunk import Chunk
from src.pipeline.artifacts import PipelineArtifacts


class PipelineBuilder:
    """
    Builds the complete document processing pipeline.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
    ) -> None:

        self.chunker = chunker
        self.embedder = embedder

    def build(self) -> PipelineArtifacts:

        documents = ingest_documents()

        chunks = self._generate_chunks(documents)

        embeddings = self.embedder.embed(chunks)

        return PipelineArtifacts(
            documents=documents,
            chunks=chunks,
            embeddings=embeddings,
        )

    def _generate_chunks(
        self,
        documents,
    ) -> list[Chunk]:

        chunks: list[Chunk] = []

        counter = 1

        for document in documents:

            document_chunks = self.chunker.chunk(document)

            for chunk in document_chunks:

                chunk.chunk_id = self._generate_chunk_id(counter)

                chunks.append(chunk)

                counter += 1

        return chunks

    @staticmethod
    def _generate_chunk_id(index: int) -> str:

        return f"chunk{index:06d}"