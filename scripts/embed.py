import json
from pathlib import Path

from src.pipeline.factory import create_pipeline_builder

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


def embed_documents():

    config = _load_config()
    embedding_model = config["embedding_model"]
    chunker = config.get("chunker", "fixed")

    # --------------------------------------------------
    # Build Pipeline
    # --------------------------------------------------

    builder = create_pipeline_builder(
        model_name=embedding_model,
        chunker=chunker,
    )

    pipeline = builder.build()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nEmbedding Summary")
    print("----------------------------")
    print(f"Documents : {len(pipeline.documents)}")
    print(f"Chunks    : {len(pipeline.chunks)}")
    print(f"Vectors   : {len(pipeline.embeddings)}")
    print(f"Model     : {builder.embedder.name}")
    print(f"Dimension : {builder.embedder.dimension}")

    return pipeline.embeddings


if __name__ == "__main__":
    embed_documents()