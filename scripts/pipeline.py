import argparse

from src.pipeline.factory import create_pipeline_builder

from src.validators.ingestion import IngestionValidator
from src.validators.chunking import ChunkingValidator
from src.validators.embedding import EmbeddingValidator
from src.validators.pipeline import PipelineValidator

from src.settings.qdrant import QdrantSettings

from src.vectordb.qdrant import QdrantVectorStore
from src.indexing.service import IndexingService

from src.retrieval.factory import (
    create_dense_retriever,
    create_bm25_retriever,
    create_hybrid_retriever,
)

from src.retrieval.pipeline import RetrievalPipeline

from src.rerankers.factory import (
    create_cross_encoder_reranker,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Research Retrieval Pipeline",
    )

    parser.add_argument(
        "--retriever",
        choices=[
            "dense",
            "bm25",
            "hybrid",
        ],
        default="dense",
        help="Retriever to use.",
    )

    parser.add_argument(
        "--reranker",
        choices=[
            "none",
            "cross",
        ],
        default="none",
        help="Optional reranker.",
    )

    parser.add_argument(
        "--query",
        default="What is PostgreSQL Full Text Search?",
        help="Search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved results.",
    )

    parser.add_argument(
        "--overwrite-index",
        action="store_true",
        help="Rebuild the vector collection.",
    )

    parser.add_argument(
        "--embedding-model",
        required=True,
        help="HuggingFace model name for sentence embeddings.",
    )

    return parser.parse_args()


def print_results(response) -> None:
    """
    Display retrieval results.
    """

    print("\nTop Results")
    print("=" * 80)

    for result in response.results:

        payload = result.payload

        title = payload.title or "Untitled"

        document = payload.document_id or "-"

        page = payload.page if payload.page is not None else "-"

        source = payload.source or "-"

        preview = payload.text.strip()

        if len(preview) > 300:

            preview = preview[:300] + "..."

        print(f"Rank      : {result.rank}")
        print(f"Score     : {result.score:.6f}")
        print(f"Document  : {document}")
        print(f"Chunk     : {result.chunk_id}")
        print(f"Title     : {title}")
        print(f"Page      : {page}")
        print(f"Source    : {source}")

        print("-" * 80)

        print(preview)

        print("=" * 80)
        print()


def main() -> None:

    args = parse_args()

    # Save configuration
    import json
    from pathlib import Path
    config_path = Path("data/metadata/config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump({"embedding_model": args.embedding_model}, f)


    # ==================================================
    # Build Pipeline
    # ==================================================

    print("\nBuilding pipeline...")

    builder = create_pipeline_builder(args.embedding_model)

    pipeline = builder.build()

    embedder = builder.embedder

    print("Pipeline built successfully.")

    # ==================================================
    # Validation
    # ==================================================

    print("\nRunning validators...")

    validators = [

        IngestionValidator(
            pipeline.documents,
        ),

        ChunkingValidator(
            pipeline.documents,
            pipeline.chunks,
        ),

        EmbeddingValidator(
            pipeline.chunks,
            pipeline.embeddings,
            embedder,
        ),

    ]

    validator = PipelineValidator(
        validators,
    )

    overall_status, report = validator.validate()

    for stage in validators:

        print(f"\n[{stage.name}]")

        for message in report[stage.name]:

            print(f"  • {message}")

    print()

    if not overall_status:

        print("Pipeline validation: FAILED")

        return

    print("Pipeline validation: PASSED")

    # ==================================================
    # Connect to Vector Store
    # ==================================================

    settings = QdrantSettings()

    print("\nConnecting to Qdrant...")

    vector_store = QdrantVectorStore(
        host=settings.host,
        port=settings.port,
    )

    if not vector_store.health_check():

        print(
            f"Unable to connect to "
            f"{vector_store.endpoint}"
        )

        return

    print("Connected successfully.")

    # ==================================================
    # Index Embeddings
    # ==================================================

    print("\nIndexing embeddings...")

    indexer = IndexingService(
        vector_store=vector_store,
        collection_name=settings.collection_name,
    )

    indexer.index(
        embeddings=pipeline.embeddings,
        overwrite=args.overwrite_index,
        batch_size=256,
    )

    print("Indexing complete.")

    # ==================================================
    # Create Retrievers
    # ==================================================

    print("\nInitializing retrievers...")

    dense_retriever = create_dense_retriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name=settings.collection_name,
    )

    bm25_retriever = create_bm25_retriever(
        chunks=pipeline.chunks,
    )

    hybrid_retriever = create_hybrid_retriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name=settings.collection_name,
        chunks=pipeline.chunks,
    )

    retrievers = {

        "dense": dense_retriever,

        "bm25": bm25_retriever,

        "hybrid": hybrid_retriever,

    }

    retriever = retrievers[
        args.retriever
    ]

    print(
        f"Retriever initialized : "
        f"{retriever.name}"
    )

    # ==================================================
    # Create Rerankers
    # ==================================================

    print("\nInitializing rerankers...")

    rerankers = {

        "none": None,

        "cross": create_cross_encoder_reranker(),

    }

    reranker = rerankers[
        args.reranker
    ]

    if reranker is None:

        print("Reranker            : None")

    else:

        print(
            f"Reranker            : "
            f"{reranker.name}"
        )

    # ==================================================
    # Create Retrieval Pipeline
    # ==================================================

    retrieval_pipeline = RetrievalPipeline(

        retriever=retriever,

        reranker=reranker,

    )

    print(
        f"Pipeline             : "
        f"{retrieval_pipeline.name}"
    )

    # ==================================================
    # Execute Retrieval
    # ==================================================

    print(
        f"\nExecuting {retrieval_pipeline.name}..."
    )

    response = retrieval_pipeline.retrieve(
        query=args.query,
        top_k=args.top_k,
    )

    # ==================================================
    # Results
    # ==================================================

    print("\nRetrieval Results")
    print("=" * 100)

    if not response.results:

        print("No results found.")

    else:

        for result in response.results:

            payload = result.payload

            title = payload.title or "Untitled"

            document = payload.document_id or "-"

            page = (
                payload.page
                if payload.page is not None
                else "-"
            )

            source = payload.source or "-"

            preview = payload.text.strip()

            if len(preview) > 300:

                preview = preview[:300] + "..."

            print(f"Rank      : {result.rank}")
            print(f"Score     : {result.score:.6f}")
            print(f"Document  : {document}")
            print(f"Chunk     : {result.chunk_id}")
            print(f"Title     : {title}")
            print(f"Page      : {page}")
            print(f"Source    : {source}")

            print("-" * 100)

            print(preview)

            print("=" * 100)
            print()

    # ==================================================
    # Pipeline Summary
    # ==================================================

    print()
    print("=" * 100)
    print("Pipeline Summary")
    print("=" * 100)

    print(
        f"{'Documents':<20}"
        f": {len(pipeline.documents)}"
    )

    print(
        f"{'Chunks':<20}"
        f": {len(pipeline.chunks)}"
    )

    print(
        f"{'Embeddings':<20}"
        f": {len(pipeline.embeddings)}"
    )

    print(
        f"{'Collection':<20}"
        f": {settings.collection_name}"
    )

    print(
        f"{'Retriever':<20}"
        f": {retriever.name}"
    )

    print(
        f"{'Reranker':<20}"
        f": {reranker.name if reranker else 'None'}"
    )

    print(
        f"{'Pipeline':<20}"
        f": {retrieval_pipeline.name}"
    )

    print(
        f"{'Query':<20}"
        f": {args.query}"
    )

    print(
        f"{'Top-K':<20}"
        f": {args.top_k}"
    )

    print(
        f"{'Returned Results':<20}"
        f": {len(response.results)}"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()