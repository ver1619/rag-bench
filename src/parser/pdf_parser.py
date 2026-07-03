from pathlib import Path

import fitz  # PyMuPDF

from src.models.document import Document


class PDFParser:
    """
    Parses PDF documents into Document objects.
    """

    def parse(self, document_id: str, pdf_path: Path) -> Document:
        """
        Parse a PDF file and return a Document object.
        """

        pdf = fitz.open(pdf_path)

        try:
            # -------------------------
            # Extract full document text
            # -------------------------
            text = ""

            for page in pdf:
                text += page.get_text()

            # -------------------------
            # Metadata
            # -------------------------
            metadata = pdf.metadata

            title = metadata.get("title")

            if not title or title.strip() == "":
                title = pdf_path.stem.replace("_", " ").replace("-", " ")

            pages = len(pdf)

            file_size = pdf_path.stat().st_size

            return Document(
                document_id=document_id,
                filename=pdf_path.name,
                title=title,
                pages=pages,
                file_size=file_size,
                text=text.strip(),
            )

        finally:
            pdf.close()