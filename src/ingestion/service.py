from src.settings.paths import (
    DOCUMENTS_DIR,
    METADATA_DIR,
)

from src.parser.pdf_parser import PDFParser
from src.utils.file_utils import (
    generate_document_id,
    list_pdf_files,
    write_metadata_json,
)

METADATA_FILE = METADATA_DIR / "documents.json"

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

    write_metadata_json(
        documents=documents,
        output_path=METADATA_FILE,
    )

    return documents