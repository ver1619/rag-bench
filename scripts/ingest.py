from src.ingestion.service import ingest_documents


def main():

    documents = ingest_documents()

    print("\nIngestion Summary")
    print("----------------------------")

    for document in documents:
        print(f"✓ {document.document_id} -> {document.filename}")

    print("\n----------------------------")
    print(f"Documents : {len(documents)}")
    print("Metadata written successfully.")

    return documents


if __name__ == "__main__":
    main()