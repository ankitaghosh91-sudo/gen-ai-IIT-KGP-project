
import argparse
import json
from pathlib import Path
from uuid import uuid4

from src.config import AppConfig
from src.data import load_saved_sample
from src.evaluation.metrics import evaluate_query
from src.pipeline.factory import _build_chunker, _build_embedder
from src.strategies import BM25Search, ChromaSearch, HybridSearch


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TOP_K = 20

K_VALUES = [1, 3, 5, 10, 20]

LABELS_PATH = Path(
    "data/evaluation/combined_relevance_labels.json"
)

OUTPUT_DIR = Path(
    "data/evaluation"
)

CHROMA_TEST_PATH = "data/processed/chroma_test"


# --------------------------------------------------
# COMMAND-LINE ARGUMENTS
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Evaluate global or document-scoped candidate retrieval."
    )

    parser.add_argument(
        "--scope",
        choices=["document", "global"],
        default="document",
        help="Retrieval scope: document or global.",
    )

    return parser.parse_args()


# --------------------------------------------------
# METRIC AGGREGATION
# --------------------------------------------------

def calculate_average_metrics(results):

    if not results:
        return {}

    metric_totals = {}

    for result in results:

        for metric_name, metric_value in result["metrics"].items():

            metric_totals[metric_name] = (
                metric_totals.get(metric_name, 0.0)
                + metric_value
            )

    return {
        metric_name: round(
            total / len(results),
            4,
        )
        for metric_name, total in metric_totals.items()
    }


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    args = parse_arguments()

    scope = args.scope

    print("\n========== CANDIDATE POOL EVALUATION ==========")
    print(f"Retrieval scope: {scope}")
    print(f"Candidate pool size: {TOP_K}")

    # Set a separate output file for each scope.
    output_path = OUTPUT_DIR / (
        f"candidate_pool_bge_{scope}.json"
    )

    # Load configuration and documents.
    config = AppConfig.from_env()

    documents = load_saved_sample()

    print(f"Documents loaded: {len(documents)}")

    if not documents:
        raise ValueError("No saved documents found.")

    # Load relevance labels.
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        all_labels = json.load(file)

    eligible_labels = [
        record
        for record in all_labels
        if record.get("relevant_chunk_ids")
    ]

    print(f"Total QA pairs: {len(all_labels)}")
    print(f"Eligible QA pairs: {len(eligible_labels)}")
    print(f"Excluded QA pairs: {len(all_labels) - len(eligible_labels)}")

    # Build the configured chunker and embedder.
    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    print(f"Chunker: {type(chunker).__name__}")
    print(f"Embedder: {type(embedder).__name__}")

    # Create a temporary Chroma collection.
    collection_name = (
        f"finance_candidate_eval_{uuid4().hex[:8]}"
    )

    dense_search = ChromaSearch(
        path=CHROMA_TEST_PATH,
        embedder=embedder,
        collection_name=collection_name,
        chunker=chunker,
    )

    try:

        # Index documents once.
        print("\nIndexing documents...")

        dense_search.index_documents(documents)

        indexed_chunks = dense_search.collection.count()

        print("Indexed chunks:", indexed_chunks)

        # Construct retrieval strategies.
        bm25_search = BM25Search(
            chunker=chunker,
        )

        hybrid_search = HybridSearch(
            dense_search=dense_search,
            sparse_search=bm25_search,
        )

        strategies = {
            "Dense": dense_search,
            "Hybrid": hybrid_search,
        }

        report = {
            "embedder": type(embedder).__name__,
            "scope": scope,
            "top_k": TOP_K,
            "k_values": K_VALUES,
            "total_documents": len(documents),
            "total_qa_pairs": len(all_labels),
            "eligible_qa_pairs": len(eligible_labels),
            "excluded_qa_pairs": len(all_labels) - len(eligible_labels),
            "indexed_chunks": indexed_chunks,
            "strategies": {},
        }

        # Evaluate each retrieval strategy.
        for strategy_name, strategy in strategies.items():

            experiment_name = (
                f"{strategy_name.lower()}_cand_"
                f"{'doc' if scope == 'document' else 'global'}_pool"
            )

            print(f"\n========== {experiment_name} ==========")

            strategy_results = []

            for index, record in enumerate(
                eligible_labels,
                start=1,
            ):

                question = record["question"]
                doc_id = record["doc_id"]

                relevant_chunk_ids = set(
                    record["relevant_chunk_ids"]
                )

                # Apply the selected retrieval scope.
                if scope == "document":

                    retrieved = strategy.search(
                        query=question,
                        documents=documents,
                        limit=TOP_K,
                        scope="document",
                        doc_id=doc_id,
                    )

                else:

                    retrieved = strategy.search(
                        query=question,
                        documents=documents,
                        limit=TOP_K,
                        scope="global",
                    )

                retrieved_chunk_ids = [
                    item.chunk_id
                    for item in retrieved
                    if item.chunk_id is not None
                ]

                # Identify source documents in the candidate pool.
                retrieved_doc_ids = list(
                    dict.fromkeys(
                        item.doc_id
                        for item in retrieved
                        if item.doc_id is not None
                    )
                )

                # Calculate retrieval metrics.
                metrics = evaluate_query(
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    relevant_chunk_ids=relevant_chunk_ids,
                    k_values=K_VALUES,
                )

                strategy_results.append(
                    {
                        "experiment": experiment_name,
                        "doc_id": doc_id,
                        "question_id": record["question_id"],
                        "question": question,
                        "relevant_chunk_ids": list(
                            relevant_chunk_ids
                        ),
                        "retrieved_chunk_ids": retrieved_chunk_ids,
                        "retrieved_doc_ids": retrieved_doc_ids,
                        "retrieved_count": len(retrieved_chunk_ids),
                        "metrics": metrics,
                    }
                )

                print(
                    f"[{index}/{len(eligible_labels)}] "
                    f"{doc_id} | "
                    f"Retrieved: {len(retrieved_chunk_ids)} | "
                    f"Source documents: {len(retrieved_doc_ids)} | "
                    f"Recall@20: {metrics['recall@20']:.4f}"
                )

            average_metrics = calculate_average_metrics(
                strategy_results
            )

            report["strategies"][experiment_name] = {
                "average_metrics": average_metrics,
                "individual_results": strategy_results,
            }

            print(f"\n{experiment_name} average metrics:")

            for metric_name, value in average_metrics.items():
                print(f"{metric_name}: {value:.4f}")

        # Save the report.
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        print("\n========== EVALUATION COMPLETED ==========")
        print(f"Scope: {scope}")
        print(f"Report saved to: {output_path}")

    finally:

        # Delete only the temporary collection.
        dense_search.client.delete_collection(
            name=collection_name
        )

        print(
            f"\nTemporary collection deleted: {collection_name}"
        )


if __name__ == "__main__":
    main()
