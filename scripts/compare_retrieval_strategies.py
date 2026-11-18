
"""Compare dense, sparse and hybrid retrieval."""

from uuid import uuid4

from src.config import AppConfig
from src.data.loader import load_saved_sample
from src.pipeline.factory import _build_chunker, _build_embedder
from src.strategies.search import ChromaSearch, BM25Search, HybridSearch


def compare_retrieval():
    config = AppConfig.from_env()
    documents = load_saved_sample()

    if not documents:
        raise ValueError("No saved documents found.")

    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    collection_name = f"finance_compare_{uuid4().hex[:8]}"

    dense_search = ChromaSearch(
        path="data/processed/chroma_test",
        embedder=embedder,
        collection_name=collection_name,
        chunker=chunker,
    )

    sparse_search = BM25Search(chunker=chunker)

    hybrid_search = HybridSearch(
        dense_search=dense_search,
        sparse_search=sparse_search,
    )

    query = (
        "What financial resources and equipment were recommended "
        "to improve research activities?"
    )

    try:
        print("Indexing documents...")
        dense_search.index_documents(documents)

        print("Stored chunks:", dense_search.collection.count())
        print("\nQuery:", query)

        strategies = {
            "DENSE (Chroma)": dense_search,
            "SPARSE (BM25)": sparse_search,
            "HYBRID (RRF)": hybrid_search,
        }

        for strategy_name, strategy in strategies.items():
            print(f"\n{'=' * 60}")
            print(strategy_name)
            print("=" * 60)

            results = strategy.search(
                query=query,
                documents=documents,
                limit=4,
            )

            for rank, item in enumerate(results, start=1):
                print(
                    f"\nRank: {rank} | "
                    f"Chunk: {item.chunk_id} | "
                    f"Score: {item.score:.6f}"
                )
                print("Document:", item.doc_id)
                print("Text:", item.text[:300].replace("\n", " "))

    finally:
        dense_search.client.delete_collection(collection_name)
        print("\nTemporary collection deleted.")


if __name__ == "__main__":
    compare_retrieval()
