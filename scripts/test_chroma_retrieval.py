
"""Smoke test: factory-configured Chroma indexing and retrieval."""

from uuid import uuid4

from src.config import AppConfig
from src.pipeline.factory import build_strategies_from_config
from src.strategies.search import ChromaSearch,KeywordSearch
from src.models.records import Document
from src.data.loader import load_saved_sample

def testChroma():
    config = AppConfig.from_env()
    config.retrieval_limit = 60

    # Use a temporary collection to avoid modifying the real collection.
    collection_name = f"finance_test_{uuid4().hex[:8]}"

    config.chroma_path = "data/processed/chroma_test"
    config.chroma_collection = collection_name

    documents = load_saved_sample()

    if not documents:
        raise ValueError("No saved documents found.")

    print("Loaded documents:", len(documents))
    print("First document:", documents[0].doc_id)
    print("QA pairs in first document:", len(documents[0].qa_pairs))

    strategies = build_strategies_from_config(config)

    # Unwrap AutoRAG -> SimpleRAG -> LimitedSearch -> ChromaSearch.
    rag = strategies.rag
    if hasattr(rag, "text"):
        rag = rag.text

    search = rag.search
    while hasattr(search, "base"):
        search = search.base

    if not isinstance(search, ChromaSearch):
        raise TypeError(
            f"Expected ChromaSearch, got {type(search).__name__}"
        )

    print("Selected embedder:", type(search.embedder).__name__)
    print("Selected chunker:", type(search.chunker).__name__)
    print("Selected Rag:", type(rag).__name__)
    if hasattr(rag, "text"):
        print("Text RAG:", type(rag.text).__name__)

    try:
        print("\nIndexing documents...")
        search.index_documents(documents)

        print("Stored chunks:", search.collection.count())

        query = documents[0].qa_pairs[0].question

        print("\nSearching:", query)

        results = strategies.rag.retrieve(query, documents)

        print("\nRetrieved evidence:")

        """
            for rank, item in enumerate(results, start=1):
                        print(f"\nRank: {rank}")
                        print("Document ID:", item.doc_id)
                        print("Score:", round(item.score, 4))
                        print("Text:", item.text)
        """
        print("\nRetrieved chunks from 175.jpeg:")

        for rank, item in enumerate(results, start=1):
            if item.doc_id == "175.jpeg":
                print(f"\nGlobal Rank: {rank}")
                print("Score:", round(item.score, 4))
                print("Text:", item.text)
                print("Chunk_id:",item.chunk_id)

    finally:
        search.client.delete_collection(collection_name)
        print("\nTemporary collection deleted.")


def testKeywordSearch():
    config = AppConfig.from_env()
    config.retrieval_limit = 60

    documents = load_saved_sample()

    if not documents:
        raise ValueError("No saved documents found.")

    print("Loaded documents:", len(documents))
    print("First document:", documents[0].doc_id)
    print("QA pairs in first document:", len(documents[0].qa_pairs))

    strategies = build_strategies_from_config(config)

    # Unwrap AutoRAG -> SimpleRAG -> LimitedSearch -> ChromaSearch.
    rag = strategies.rag
    if hasattr(rag, "text"):
        rag = rag.text

    search = rag.search
    while hasattr(search, "base"):
        search = search.base

    if not isinstance(search, KeywordSearch):
        raise TypeError(
            f"Expected KeywordSearch, got {type(search).__name__}"
        )

    print("Selected search:", type(search).__name__)
    print("Selected chunker:", type(search.chunker).__name__)
    print("Selected Rag:", type(rag).__name__)
    if hasattr(rag, "text"):
        print("Text RAG:", type(rag.text).__name__)

       
    query = documents[0].qa_pairs[0].question

    print("\nSearching:", query)

    results = strategies.rag.retrieve(query, documents)

    print("\nRetrieved evidence:")

    """
        for rank, item in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print("Document ID:", item.doc_id)
            print("Score:", round(item.score, 4))
            print("Text:", item.text)
    """
    print("\nRetrieved chunks from 175.jpeg:")

    for rank, item in enumerate(results, start=1):
        if item.doc_id == "175.jpeg":
            print(f"\nGlobal Rank: {rank}")
            print("Score:", round(item.score, 4))
            print("Text:", item.text)
            print("Chunk_id:",item.chunk_id)

   
if __name__ == "__main__":
    testChroma()
   #testKeywordSearch()
