
"""Build a persistent ChromaDB collection for the 500-document sample."""
import argparse

from src.config import AppConfig
from src.pipeline.factory import _build_chunker, _build_embedder
from src.strategies.search import ChromaSearch
from src.data.loader import load_saved_sample


INPUT_PATH = "data/sample_500/documents.json"
DEFAULT_COLLECTION_NAME = "finance_sample_500"


def index_before():
    config = AppConfig.from_env()

    print("\n========== BUILDING 500-DOCUMENT CHROMA INDEX ==========")
    print("Input:", INPUT_PATH)
    print("Chroma path:", config.chroma_path)
    print("Collection:", DEFAULT_COLLECTION_NAME)
    print("Chunking strategy:", config.chunking_strategy)
    print("Chunk size:", config.chunk_size)
    print("Chunk overlap:", config.chunk_overlap)
    print("Embedder strategy:", config.embedder_strategy)
    print("Embedding model:", config.embedding_model_name)

    documents = load_saved_sample(path=INPUT_PATH)

    print("\nDocuments loaded:", len(documents))

    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    search = ChromaSearch(
        path=config.chroma_path,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedder=embedder,
        chunker=chunker,
    )

    existing_count = search.collection.count()

    if existing_count > 0:
        raise RuntimeError(
            f"Collection '{DEFAULT_COLLECTION_NAME}' already contains "
            f"{existing_count} chunks. Refusing to overwrite it. "
            "Use a new collection name if you intend to rebuild."
        )

    search.index_documents(documents)

    print("\n========== INDEXING COMPLETED ==========")
    print("Documents indexed:", len(documents))
    print("Stored chunks:", search.collection.count())
    print("Collection:", search.collection.name)
    print("Chroma path:", config.chroma_path)


def index_after():
    parser = argparse.ArgumentParser(
        description="Build a persistent ChromaDB collection for 500 documents."
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Name of the ChromaDB collection.",
    )

    args = parser.parse_args()
    collection_name = args.collection

    config = AppConfig.from_env()

    print("\n========== BUILDING 500-DOCUMENT CHROMA INDEX ==========")
    print("Input:", INPUT_PATH)
    print("Chroma path:", config.chroma_path)
    print("Collection:", collection_name)
    print("Chunking strategy:", config.chunking_strategy)
    print("Chunk size:", config.chunk_size)
    print("Chunk overlap:", config.chunk_overlap)
    print("Embedder strategy:", config.embedder_strategy)
    print("Embedding model:", config.embedding_model_name)

    documents = load_saved_sample(path=INPUT_PATH)

    print("\nDocuments loaded:", len(documents))

    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    search = ChromaSearch(
        path=config.chroma_path,
        collection_name=collection_name,
        embedder=embedder,
        chunker=chunker,
    )

    existing_count = search.collection.count()

    if existing_count > 0:
        raise RuntimeError(
            f"Collection '{collection_name}' already contains "
            f"{existing_count} chunks. Refusing to overwrite it. "
            "Use a new collection name if you intend to rebuild."
        )

    search.index_documents(documents)

    print("\n========== INDEXING COMPLETED ==========")
    print("Documents indexed:", len(documents))
    print("Stored chunks:", search.collection.count())
    print("Collection:", search.collection.name)
    print("Chroma path:", config.chroma_path)


if __name__ == "__main__":
    #index_before()
    index_after()
