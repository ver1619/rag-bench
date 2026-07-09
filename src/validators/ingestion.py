from src.models.document import Document
from src.validators.base import BaseValidator


class IngestionValidator(BaseValidator):
    """
    Validates ingested documents.
    """

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    @property
    def name(self) -> str:
        return "Ingestion"

    def validate(self) -> tuple[bool, list[str]]:

        messages: list[str] = []
        passed = True

        if not self.documents:
            return False, ["No documents were ingested."]

        messages.append(f"Documents found: {len(self.documents)}")

        # ------------------------------------------------------------------
        # Unique Document IDs
        # ------------------------------------------------------------------

        document_ids = [doc.document_id for doc in self.documents]

        if len(document_ids) != len(set(document_ids)):
            passed = False
            messages.append("Duplicate document IDs detected.")
        else:
            messages.append("Document IDs are unique.")

        # ------------------------------------------------------------------
        # Empty Documents
        # ------------------------------------------------------------------

        empty_documents = [
            doc.filename
            for doc in self.documents
            if not doc.text.strip()
        ]

        if empty_documents:
            passed = False
            messages.append(
                f"Empty documents: {len(empty_documents)}"
            )
        else:
            messages.append("No empty documents found.")

        # ------------------------------------------------------------------
        # Page Validation
        # ------------------------------------------------------------------

        invalid_pages = [
            doc.filename
            for doc in self.documents
            if doc.pages <= 0
        ]

        if invalid_pages:
            passed = False
            messages.append(
                f"Invalid page counts: {len(invalid_pages)}"
            )
        else:
            messages.append("Page counts are valid.")

        # ------------------------------------------------------------------
        # File Size Validation
        # ------------------------------------------------------------------

        invalid_sizes = [
            doc.filename
            for doc in self.documents
            if doc.file_size <= 0
        ]

        if invalid_sizes:
            passed = False
            messages.append(
                f"Invalid file sizes: {len(invalid_sizes)}"
            )
        else:
            messages.append("File sizes are valid.")

        # ------------------------------------------------------------------
        # Title Validation
        # ------------------------------------------------------------------

        missing_titles = [
            doc.filename
            for doc in self.documents
            if not doc.title.strip()
        ]

        if missing_titles:
            passed = False
            messages.append(
                f"Missing titles: {len(missing_titles)}"
            )
        else:
            messages.append("Titles are present.")

        return passed, messages