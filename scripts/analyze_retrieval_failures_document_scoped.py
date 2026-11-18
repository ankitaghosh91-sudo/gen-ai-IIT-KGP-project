
"""Analyze retrieval failures against the relevance ground truth.

Supported strategies:
    - dense
    - hybrid

The script saves QA pairs where at least one of the top-k retrieved
chunks is not present in the ground-truth relevant_chunk_ids.
"""

import json
from pathlib import Path
from uuid import uuid4

from src.config import AppConfig
from src.data.loader import load_saved_sample
from src.pipeline.factory import _build_chunker, _build_embedder
from src.strategies.search import ChromaSearch, BM25Search, HybridSearch


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

STRATEGY = "hybrid"
#STRATEGY = "dense"  # Change to "dense" to analyze dense retrieval
TOP_K = 5
scope = "document"

OUTPUT_PATH = Path(
    "data/evaluation/bge_hybrid_failure_analysis_document_scoped.json"
)

LABELS_PATH = Path(
    "data/evaluation/combined_relevance_labels.json"
)


def analyze_retrieval_failures():

    if STRATEGY not in {"dense", "hybrid"}:
        raise ValueError(
            "STRATEGY must be either 'dense' or 'hybrid'."
        )

    # 1. Load configuration and documents
    config = AppConfig.from_env()
    documents = load_saved_sample()

    if not documents:
        raise ValueError("No saved documents found.")

    # 2. Load relevance ground truth
    with LABELS_PATH.open(encoding="utf-8") as f:
        labels = json.load(f)

    eligible_labels = [
        record
        for record in labels
        if record.get("relevant_chunk_ids")
    ]

    excluded_count = len(labels) - len(eligible_labels)

    print("\nDataset summary")
    print("-" * 50)
    print("Total QA pairs:", len(labels))
    print("Eligible QA pairs:", len(eligible_labels))
    print("Excluded QA pairs:", excluded_count)
    print("Documents loaded:", len(documents))

    # 3. Build chunker and embedder
    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    print("\nChunker:", chunker)
    print("Embedder:", embedder.model_name)
    print("Retrieval strategy:", STRATEGY)
    print("Top K:", TOP_K)

    # 4. Create a temporary Chroma collection
    collection_name = f"finance_failure_{uuid4().hex[:8]}"

    dense_search = ChromaSearch(
        path="data/processed/chroma_test",
        embedder=embedder,
        collection_name=collection_name,
        chunker=chunker,
    )

    # 5. Build the selected retrieval strategy
    if STRATEGY == "dense":
        search_strategy = dense_search

    else:
        sparse_search = BM25Search(chunker=chunker)

        search_strategy = HybridSearch(
            dense_search=dense_search,
            sparse_search=sparse_search,
        )

    failure_records = []

    try:

        # 6. Index documents
        print("\nIndexing documents into temporary collection...")

        dense_search.index_documents(documents)

        print(
            "Indexed chunks:",
            dense_search.collection.count(),
        )

        # 7. Analyze each eligible QA pair
        for index, record in enumerate(
            eligible_labels,
            start=1,
        ):

            question = record["question"]
            doc_id = record["doc_id"]

            relevant_ids = set(
                record["relevant_chunk_ids"]
            )

            print(
                f"\nAnalyzing QA pair "
                f"{index}/{len(eligible_labels)} "
            )
            print("Document:", doc_id)
            print("Question:", question)

            # Retrieve top-k chunks
            results = search_strategy.search(
                query=question,
                documents=documents,
                limit=TOP_K,
                scope=scope,
                doc_id=(
                    doc_id
                    if scope == "document"
                    else None
                ),
            )

            retrieved_chunks = []

            for rank, item in enumerate(results, start=1):

                chunk_id = item.chunk_id

                retrieved_chunks.append({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "doc_id": item.doc_id,
                    "score": item.score,
                    "text": item.text,
                    "is_relevant": chunk_id in relevant_ids,
                })

            # Separate relevant and unexpected retrievals
            retrieved_relevant = [
                item
                for item in retrieved_chunks
                if item["is_relevant"]
            ]

            retrieved_unlabelled = [
                item
                for item in retrieved_chunks
                if not item["is_relevant"]
            ]

            retrieved_relevant_ids = {
                item["chunk_id"]
                for item in retrieved_chunks
                if item["chunk_id"] in relevant_ids
            }

            missed_relevant_ids = (
                relevant_ids - retrieved_relevant_ids
            )

            # Complete retrieval failure:
            # No retrieved chunk matches any ground-truth relevant chunk.
            if not retrieved_relevant_ids:

                failure_records.append({

                    # Original QA information
                    "doc_id": record["doc_id"],
                    "question_id": record["question_id"],
                    "question": question,
                    "gold_answer": record.get("gold_answer"),

                    # Ground truth
                    "relevant_chunk_ids": sorted(relevant_ids),

                    # Retrieval results
                    "retrieved_chunks": retrieved_chunks,

                    "retrieved_relevant_chunk_ids": sorted(
                        retrieved_relevant_ids
                    ),

                    "missed_relevant_chunk_ids": sorted(
                        missed_relevant_ids
                    ),
                    "failure_type": "complete_retrieval_failure",

                    "retrieved_count": len(retrieved_chunks),
                    "relevant_retrieved_count": len(
                        retrieved_relevant
                    ),
                })

        # 8. Save failure analysis
        report = {
            "analysis_summary": {
                "total_qa_pairs": len(labels),
                "eligible_qa_pairs": len(eligible_labels),
                "excluded_qa_pairs": excluded_count,
                "analyzed_qa_pairs": len(eligible_labels),
                "failed_qa_pairs": len(failure_records),
                "clean_qa_pairs": (
                    len(eligible_labels) - len(failure_records)
                ),
                "retrieval_strategy": STRATEGY,
                "embedder": embedder.model_name,
                "top_k": TOP_K,
            },
            "failure_records": failure_records,
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False,
            )

        # 9. Print summary
        print("\n" + "=" * 60)
        print("RETRIEVAL FAILURE ANALYSIS COMPLETED")
        print("=" * 60)

        print("Strategy:", STRATEGY)
        print("Total eligible QA pairs:", len(eligible_labels))
        print("Complete retrieval failures:", len(failure_records))
        print(
            "QA pairs with at least one relevant chunk retrieved:",
            len(eligible_labels) - len(failure_records),
        )

        print("\nOutput saved to:", OUTPUT_PATH)

    finally:

        # 10. Clean up temporary Chroma collection
        dense_search.client.delete_collection(collection_name)

        print("\nTemporary Chroma collection deleted.")


if __name__ == "__main__":
    analyze_retrieval_failures()
