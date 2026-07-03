from pathlib import Path

from src.parser.pdf_parser import PDFParser
from src.utils.file_utils import (
    generate_document_id,
    list_pdf_files,
    write_metadata_json,
)


DATA_DIR = Path("data")
DOCUMENTS_DIR = DATA_DIR / "documents"
METADATA_FILE = DATA_DIR / "metadata" / "documents.json"


def ingest_documents():
    """
    Ingest every PDF document into Document objects.
    """

    parser = PDFParser()

    pdf_files = list_pdf_files(DOCUMENTS_DIR)

    documents = []

    for index, pdf_file in enumerate(pdf_files, start=1):

        document = parser.parse(
            document_id=generate_document_id(index),
            pdf_path=pdf_file,
        )

        documents.append(document)

        print(f"✓ {document.document_id} -> {document.filename}")

    write_metadata_json(
        documents=documents,
        output_path=METADATA_FILE,
    )

    print(f"\nProcessed {len(documents)} document(s).")
    print(f"Metadata saved to {METADATA_FILE}")

    return documents


if __name__ == "__main__":
    ingest_documents()