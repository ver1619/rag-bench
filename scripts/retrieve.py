import json
from pathlib import Path

from src.pipeline.factory import create_pipeline_builder
from src.ingestion.service import ingest_documents

from src.retrieval.factory import (
    create_dense_retriever,
    create_bm25_retriever,
    create_hybrid_retriever,
)

from src.settings.qdrant import QdrantSettings
from src.vectordb.qdrant import QdrantVectorStore


# ==================================================
# Configuration
# ==================================================

RETRIEVER = "hybrid"
QUERY = "What is PostgreSQL Full Text Search?"
TOP_K = 5

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


def retrieve():

    # ==================================================
    # Load Configuration
    # ==================================================

    embedding_model = _load_embedding_model()

    # ==================================================
    # Prepare Embedder and Chunks
    # ==================================================

    print("\nPreparing retrieval components...")

    builder = create_pipeline_builder(embedding_model)

    embedder = builder.embedder

    documents = ingest_documents()
    chunks = builder._generate_chunks(documents)

    print("Components ready.")

    # ==================================================
    # Vector Store
    # ==================================================

    settings = QdrantSettings()

    vector_store = QdrantVectorStore(
        host=settings.host,
        port=settings.port,
    )

    if not vector_store.health_check():

        print(
            f"Unable to connect to {vector_store.endpoint}"
        )

        return

    # ==================================================
    # Create Retrievers
    # ==================================================

    dense = create_dense_retriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name=settings.collection_name,
    )

    bm25 = create_bm25_retriever(
        chunks=chunks,
    )

    hybrid = create_hybrid_retriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name=settings.collection_name,
        chunks=chunks,
    )

    # ==================================================
    # Select Retriever
    # ==================================================

    retrievers = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": hybrid,
    }

    if RETRIEVER not in retrievers:

        raise ValueError(
            f"Unknown retriever '{RETRIEVER}'. "
            f"Available: {list(retrievers.keys())}"
        )

    retriever = retrievers[RETRIEVER]

    print(f"\nUsing: {retriever.name}")

    # ==================================================
    # Retrieval
    # ==================================================

    response = retriever.retrieve(
        query=QUERY,
        top_k=TOP_K,
    )

    # ==================================================
    # Results
    # ==================================================

    print("\nTop Results")
    print("-" * 60)

    for result in response.results:

        print(
            f"[{result.rank}] "
            f"{result.chunk_id:<15} "
            f"Score = {result.score:.6f}"
        )

    # ==================================================
    # Summary
    # ==================================================

    print("\n" + "=" * 60)
    print("Retrieval Summary")
    print("=" * 60)

    print(f"Retriever : {retriever.name}")
    print(f"Query     : {QUERY}")
    print(f"Results   : {len(response.results)}")

    print("=" * 60)

    return response


if __name__ == "__main__":
    retrieve()