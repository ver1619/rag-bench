from dataclasses import dataclass


@dataclass(frozen=True)
class QdrantSettings:
    """
    Qdrant infrastructure settings.
    """

    host: str = "localhost"
    port: int = 6333
    collection_name: str = "research"