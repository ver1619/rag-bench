import json
from pathlib import Path

from src.pipeline.factory import create_pipeline_builder
from src.ingestion.service import ingest_documents

CONFIG_PATH = Path("data/metadata/config.json")


def _load_config() -> dict:
    """
    Load the pipeline config.
    """

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            "Configuration file not found. "
            "Run pipeline.py first."
        )

    with CONFIG_PATH.open("r") as f:
        config = json.load(f)

    return config


def generate_chunk_id(index: int) -> str:
    """
    Generate a globally unique chunk ID.
    """
    return f"chunk{index:06d}"


def chunk_documents():
    """
    Chunk all ingested documents using the configured chunker.
    """

    config = _load_config()
    embedding_model = config["embedding_model"]
    chunker_name = config.get("chunker", "fixed")

    documents = ingest_documents()

    builder = create_pipeline_builder(
        model_name=embedding_model,
        chunker=chunker_name,
    )

    chunker = builder.chunker

    chunks = []

    chunk_counter = 1

    for document in documents:

        document_chunks = chunker.chunk(document)

        for chunk in document_chunks:

            chunk.chunk_id = generate_chunk_id(chunk_counter)

            chunks.append(chunk)

            chunk_counter += 1

    print(f"\nProcessed {len(documents)} documents.")
    print(f"Generated {len(chunks)} chunks.")

    return chunks


if __name__ == "__main__":
    chunk_documents()