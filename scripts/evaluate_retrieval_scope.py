
import json
from pathlib import Path
from uuid import uuid4

from src.config import AppConfig
from src.data.loader import load_saved_sample
from src.pipeline.factory import _build_chunker, _build_embedder
from src.evaluation.metrics import evaluate_query

from src.strategies.search import (
    ChromaSearch,
    BM25Search,
    HybridSearch,
)


def evaluate_retrieval_scope():

    # 1. Load configuration and documents
    config = AppConfig.from_env()
    documents = load_saved_sample()

    if not documents:
        raise ValueError("No saved documents found.")

    # 2. Load relevance ground truth
    labels_path = Path(
        "data/evaluation/combined_relevance_labels.json"
    )

    with labels_path.open(encoding="utf-8") as f:
        labels = json.load(f)

    eligible_labels = [
        record
        for record in labels
        if record["relevant_chunk_ids"]
    ]

    excluded_labels = [
        record
        for record in labels
        if not record["relevant_chunk_ids"]
    ]

    print("\nEvaluation dataset:")
    print("Total QA pairs:", len(labels))
    print("Eligible QA pairs:", len(eligible_labels))
    print("Excluded QA pairs:", len(excluded_labels))

    # 3. Build shared chunker and embedder
    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    print("Chunker:", chunker)
    print("Embedder model:", embedder.model_name)

    # 4. Create a temporary Chroma collection
    collection_name = f"finance_scope_eval_{uuid4().hex[:8]}"

    dense_search = ChromaSearch(
        path="data/processed/chroma_test",
        embedder=embedder,
        collection_name=collection_name,
        chunker=chunker,
    )

    # 5. Create Hybrid search
    sparse_search = BM25Search(chunker=chunker)

    hybrid_search = HybridSearch(
        dense_search=dense_search,
        sparse_search=sparse_search,
    )

    # Four configurations:
    configurations = {
        "Dense - Global": (dense_search, "global"),
        "Hybrid - Global": (hybrid_search, "global"),
        "Dense - Document": (dense_search, "document"),
        "Hybrid - Document": (hybrid_search, "document"),
    }

    print("\nConfigurations:")
    for name in configurations:
        print("-", name)

    try:

        # 6. Index documents once
        print("\nIndexing documents...")

        dense_search.index_documents(documents)

        print(
            "Indexed chunks:",
            dense_search.collection.count(),
        )

        # 7. Prepare result containers
        evaluation_results = {
            name: []
            for name in configurations
        }

        # 8. Evaluate all eligible QA pairs
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
                f"\nEvaluating QA pair "
                f"{index}/{len(eligible_labels)}"
            )

            print("Document:", doc_id)
            print("Question:", question)

            # Run all four configurations
            for configuration_name, (
                strategy,
                scope,
            ) in configurations.items():

                results = strategy.search(
                    query=question,
                    documents=documents,
                    limit=5,
                    scope=scope,
                    doc_id=(
                        doc_id
                        if scope == "document"
                        else None
                    ),
                )

                retrieved_ids = [
                    item.chunk_id
                    for item in results
                    if item.chunk_id is not None
                ]

                metrics = evaluate_query(
                    retrieved_chunk_ids=retrieved_ids,
                    relevant_chunk_ids=relevant_ids,
                )

                evaluation_results[
                    configuration_name
                ].append({
                    "question_number": index,
                    "doc_id": doc_id,
                    "question_id": record["question_id"],
                    "question": question,
                    "scope": scope,
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "retrieved_chunk_ids": retrieved_ids,
                    "metrics": metrics,
                })

                print(
                    f"{configuration_name}: "
                    f"retrieved={retrieved_ids}, "
                    f"MRR@5={metrics['mrr@5']:.4f}"
                )

        print("\nEvaluation completed.")

        # 9. Aggregate metrics
        metric_names = [
            "precision@1",
            "precision@3",
            "precision@5",
            "recall@1",
            "recall@3",
            "recall@5",
            "mrr@1",
            "mrr@3",
            "mrr@5",
        ]

        averages = {}

        for configuration_name, records in (
            evaluation_results.items()
        ):

            averages[configuration_name] = {
                metric: sum(
                    record["metrics"][metric]
                    for record in records
                ) / len(records)
                for metric in metric_names
            }

        # 10. Print comparison
        print("\nAverage Evaluation Metrics")
        print("-" * 100)

        print(
            f"{'Metric':<15}"
            f"{'Dense Global':>18}"
            f"{'Hybrid Global':>18}"
            f"{'Dense Document':>20}"
            f"{'Hybrid Document':>20}"
        )

        print("-" * 100)

        for metric in metric_names:

            print(
                f"{metric:<15}"
                f"{averages['Dense - Global'][metric]:>18.4f}"
                f"{averages['Hybrid - Global'][metric]:>18.4f}"
                f"{averages['Dense - Document'][metric]:>20.4f}"
                f"{averages['Hybrid - Document'][metric]:>20.4f}"
            )

        print("-" * 100)

        # 11. Save to a separate report
        output_path = Path(
            "data/evaluation/retrieval_scope_comparison_bge.json"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "embedder": embedder.model_name,
            "total_qa_pairs": len(labels),
            "eligible_qa_pairs": len(eligible_labels),
            "excluded_qa_pairs": len(excluded_labels),
            "top_k": 5,
            "evaluation_results": evaluation_results,
            "averages": averages,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                indent=4,
                ensure_ascii=False,
            )

        print("\nEvaluation report saved to:", output_path)

    finally:

        dense_search.client.delete_collection(
            collection_name
        )

        print("Temporary Chroma collection deleted.")


if __name__ == "__main__":
    evaluate_retrieval_scope()
