import argparse
import json
import time
from pathlib import Path

from src.pipeline.factory import create_pipeline_builder
from src.ingestion.service import ingest_documents

from src.settings.qdrant import QdrantSettings
from src.vectordb.qdrant import QdrantVectorStore

from src.retrieval.factory import (
    create_dense_retriever,
    create_bm25_retriever,
    create_hybrid_retriever,
)

from src.retrieval.pipeline import RetrievalPipeline

from src.rerankers.factory import (
    create_cross_encoder_reranker,
    create_bge_reranker,
    create_mxbai_reranker,
)

from src.evaluation.dataset import EvaluationDataset
from src.evaluation.latency import Latency
from src.evaluation.runner import EvaluationRunner

from src.evaluation.metrics import (
    RecallAtK,
    PrecisionAtK,
    MeanReciprocalRank,
    NormalizedDiscountedCumulativeGain,
)


def parse_args() -> argparse.Namespace:
    """
    Parse benchmark arguments.
    """

    parser = argparse.ArgumentParser(
        description="Retrieval Benchmark",
    )

    parser.add_argument(
        "--retriever",
        choices=[
            "dense",
            "bm25",
            "hybrid",
        ],
        default="dense",
        help="Retriever.",
    )

    parser.add_argument(
        "--reranker",
        choices=[
            "none",
            "cross",
            "bge",
            "mxbai",
        ],
        default="none",
        help="Optional reranker.",
    )

    parser.add_argument(
        "--dataset",
        default="data/queries/queries.json",
        help="Evaluation dataset.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N queries.",
    )

    return parser.parse_args()


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


def main():

    args = parse_args()

    benchmark_start = time.perf_counter()

    config = _load_config()
    embedding_model = config["embedding_model"]
    chunker = config.get("chunker", "fixed")

    print("\nBuilding retrieval pipeline...")

    builder = create_pipeline_builder(
        model_name=embedding_model,
        chunker=chunker,
    )

    print("Ingesting and chunking documents...")
    documents = ingest_documents()
    chunks = builder._generate_chunks(documents)

    embedder = builder.embedder

    settings = QdrantSettings()

    vector_store = QdrantVectorStore(
        host=settings.host,
        port=settings.port,
    )

    if not vector_store.health_check():

        print("Unable to connect to Qdrant.")

        return

    retrievers = {

        "dense": create_dense_retriever(
            vector_store=vector_store,
            embedder=embedder,
            collection_name=settings.collection_name,
        ),

        "bm25": create_bm25_retriever(
            chunks=chunks,
        ),

        "hybrid": create_hybrid_retriever(
            vector_store=vector_store,
            embedder=embedder,
            collection_name=settings.collection_name,
            chunks=chunks,
        ),

    }

    rerankers = {

        "none": None,

        "cross": create_cross_encoder_reranker(),

        "bge": create_bge_reranker(),

        "mxbai": create_mxbai_reranker(),

    }

    retrieval_pipeline = RetrievalPipeline(

        retriever=retrievers[args.retriever],

        reranker=rerankers[args.reranker],

    )

    dataset = EvaluationDataset(
        args.dataset,
    )

    # ==================================================
    # Evaluation Metrics
    # ==================================================

    metrics = [

        RecallAtK(),

        PrecisionAtK(),

        MeanReciprocalRank(),

        NormalizedDiscountedCumulativeGain(),

    ]

    # ==================================================
    # Evaluation Runner
    # ==================================================

    runner = EvaluationRunner(

        retrieval_pipeline=retrieval_pipeline,

        dataset=dataset,

        metrics=metrics,

        latency=Latency(),

    )

    print("\nStarting benchmark...")
    print("-" * 70)

    print(
        f"Pipeline : "
        f"{retrieval_pipeline.name}"
    )

    print(
        f"Dataset  : "
        f"{args.dataset}"
    )

    print(
        f"Top-K    : "
        f"{args.top_k}"
    )

    print(
        f"Limit    : "
        f"{args.limit if args.limit is not None else 'All'}"
    )

    print("-" * 70)

    benchmark = runner.run(

        top_k=args.top_k,

        limit=args.limit,

        show_progress=True,

    )

    benchmark_elapsed = (
        time.perf_counter()
        - benchmark_start
    )

    summary = benchmark.summary

    # ==================================================
    # Benchmark Summary
    # ==================================================

    print()
    print("=" * 80)
    print("Retrieval Benchmark Summary")
    print("=" * 80)

    print(
        f"{'Pipeline':<24}: "
        f"{retrieval_pipeline.name}"
    )

    print(
        f"{'Retriever':<24}: "
        f"{retrieval_pipeline.retriever.name}"
    )

    print(
        f"{'Reranker':<24}: "
        f"{retrieval_pipeline.reranker.name if retrieval_pipeline.reranker else 'None'}"
    )

    print(
        f"{'Dataset':<24}: "
        f"{args.dataset}"
    )

    print(
        f"{'Queries Evaluated':<24}: "
        f"{summary.queries}"
    )

    print(
        f"{'Top-K':<24}: "
        f"{args.top_k}"
    )

    print("-" * 80)

    print(
        f"{'Recall@K':<24}: "
        f"{summary.recall:.4f}"
    )

    print(
        f"{'Precision@K':<24}: "
        f"{summary.precision:.4f}"
    )

    print(
        f"{'MRR':<24}: "
        f"{summary.mrr:.4f}"
    )

    print(
        f"{'nDCG@K':<24}: "
        f"{summary.ndcg:.4f}"
    )

    print(
        f"{'Latency (ms)':<24}: "
        f"{summary.latency_ms:.2f}"
    )

    print("-" * 80)

    print(
        f"{'Benchmark Time':<24}: "
        f"{benchmark_elapsed:.2f} sec"
    )

    print("=" * 80)

    # ==================================================
    # Per-query Statistics
    # ==================================================

    print()

    print(
        f"Completed "
        f"{summary.queries} "
        f"evaluation queries."
    )

    print(
        f"Collected "
        f"{len(benchmark.results)} "
        f"per-query benchmark results."
    )


if __name__ == "__main__":

    main()
