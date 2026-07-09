from src.retrieval.models import RetrievalResponse


class DocumentMapper:
    """
    Maps retrieved chunks into an ordered list of
    unique document identifiers.

    Duplicate document IDs are removed while preserving
    the rank of the first retrieved chunk.
    """

    def map(
        self,
        response: RetrievalResponse,
    ) -> list[str]:

        seen: set[str] = set()

        documents: list[str] = []

        for result in response.results:

            document_id = result.payload.document_id

            if document_id is None:

                continue

            if document_id in seen:

                continue

            seen.add(document_id)

            documents.append(
                document_id
            )

        return documents