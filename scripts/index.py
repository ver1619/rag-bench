import json
from pathlib import Path

from src.pipeline.factory import create_pipeline_builder

from src.indexing.service import IndexingService
from src.settings.qdrant import QdrantSettings
from src.vectordb.qdrant import QdrantVectorStore

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


def index_documents():

    embedding_model = _load_embedding_model()

    # --------------------------------------------------
    # Build Pipeline
    # --------------------------------------------------

    builder = create_pipeline_builder(embedding_model)

    pipeline = builder.build()

    # --------------------------------------------------
    # Vector Store
    # --------------------------------------------------

    settings = QdrantSettings()

    vector_store = QdrantVectorStore(
        host=settings.host,
        port=settings.port,
    )

    if not vector_store.health_check():

        print(
            f"Unable to connect to Qdrant at "
            f"{vector_store.endpoint}"
        )
        return

    # --------------------------------------------------
    # Indexing
    # --------------------------------------------------

    indexer = IndexingService(
        vector_store=vector_store,
        collection_name=settings.collection_name,
    )

    indexer.index(
        embeddings=pipeline.embeddings,
        overwrite=True,
        batch_size=256,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nIndexing Summary")
    print("----------------------------")
    print(f"Documents  : {len(pipeline.documents)}")
    print(f"Chunks     : {len(pipeline.chunks)}")
    print(f"Embeddings : {len(pipeline.embeddings)}")
    print(f"Collection : {settings.collection_name}")
    print(f"Vector DB  : {vector_store.name}")

    return pipeline


if __name__ == "__main__":
    index_documents()