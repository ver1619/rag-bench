import json
from pathlib import Path

from src.models.document import Document


def list_pdf_files(directory: Path) -> list[Path]:
    """
    Return all PDF files in the directory sorted alphabetically.
    """

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(directory.glob("*.pdf"))


def generate_document_id(index: int) -> str:
    """
    Generate a sequential document ID.

    Example:
        1 -> doc001
        12 -> doc012
        123 -> doc123
    """

    return f"doc{index:03d}"


def write_metadata_json(
    documents: list[Document],
    output_path: Path,
) -> None:
    """
    Write lightweight document metadata to JSON.
    """

    metadata = []

    for document in documents:
        metadata.append(
            {
                "document_id": document.document_id,
                "filename": document.filename,
                "title": document.title,
                "pages": document.pages,
                "file_size": document.file_size,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)