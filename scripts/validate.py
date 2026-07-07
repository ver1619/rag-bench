import json
from pathlib import Path

from src.pipeline.factory import create_pipeline_builder

from src.validators.ingestion import IngestionValidator
from src.validators.chunking import ChunkingValidator
from src.validators.embedding import EmbeddingValidator
from src.validators.pipeline import PipelineValidator

CONFIG_PATH = Path("data/metadata/config.json")


def _load_embedding_model() -> str:
    """
    Load the embedding model name from the pipeline config.
    """

    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            "Configuration file not found. "
            "Run pipeline.py first."
        )

    with CONFIG_PATH.open("r") as f:
        config = json.load(f)

    return config["embedding_model"]


def main():

    embedding_model = _load_embedding_model()

    builder = create_pipeline_builder(embedding_model)

    pipeline = builder.build()

    validators = [

        IngestionValidator(
            pipeline.documents
        ),

        ChunkingValidator(
            pipeline.documents,
            pipeline.chunks,
        ),

        EmbeddingValidator(
            pipeline.chunks,
            pipeline.embeddings,
            builder.embedder,
        ),

    ]

    validator = PipelineValidator(validators)

    overall_status, report = validator.validate()

    print("\n" + "=" * 60)
    print(" Retrieval Benchmark Validation")
    print("=" * 60)

    for stage in validators:

        print(f"\n[{stage.name}]")

        for message in report[stage.name]:
            print(f"  • {message}")

    print("\n" + "=" * 60)

    print(
        "Overall Status : "
        + ("PASS" if overall_status else "FAIL")
    )

    print("=" * 60)


if __name__ == "__main__":
    main()