import argparse
import json
import logging
from pathlib import Path

from src.models.records import Document
from src.strategies.search import ChromaSearch, BM25Search, HybridSearch
from src.strategies.reranker import BGEReranker
from src.evaluation.metrics import evaluate_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(docs_path: str) -> list[Document]:
    """Load documents JSON file and convert to Document instances."""
    path = Path(docs_path)
    if not path.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    
    return [
        Document(
            doc_id=doc.get("doc_id"),
            content=doc.get("content", ""),
            qa_pairs=doc.get("qa_pairs", []),
            image_path=doc.get("image_path", None),
        )
        for doc in raw_docs
    ]

def load_labels(labels_path: str) -> list[dict]:
    """Load evaluation labels / gold set JSON file."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval_strategies(
    docs_path: str,
    labels_path: str,
    output_path: str,
):
    """Run full evaluation across retrieval strategies including the BGE cross-encoder reranker."""
    logger.info("Loading documents from %s", docs_path)
    documents = load_documents(docs_path)

    logger.info("Loading labels from %s", labels_path)
    labels = load_labels(labels_path)

    # Filter out records missing necessary fields (supporting 'relevant_chunk_id' from your schema)
    eligible_labels = []
    for item in labels:
        q = item.get("question")
        chunk_ids = item.get("relevant_chunk_id") or item.get("chunk_ids") or item.get("relevant_chunk_ids")
        if q and chunk_ids:
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            item["_normalized_chunk_ids"] = [str(c).strip() for c in chunk_ids]
            eligible_labels.append(item)

    logger.info("Found %d eligible QA pairs for evaluation.", len(eligible_labels))

    if len(eligible_labels) == 0:
        logger.error("No eligible QA pairs found! Check your label key matching.")
        return

    # Initialize Search Strategies
    logger.info("Initializing search strategies...")
    dense_search = ChromaSearch()
    dense_search.index_documents(documents)

    sparse_search = BM25Search()
    hybrid_search = HybridSearch(dense_search=dense_search, sparse_search=sparse_search)

    # Initialize Local Free Reranker
    reranker = BGEReranker(model_name="BAAI/bge-reranker-base")

    strategy_names = [
        "Dense (Chroma)",
        "Sparse (BM25)",
        "Hybrid + BGE Reranker",
    ]
    evaluation_results = {name: [] for name in strategy_names}

    # Run evaluation loop
    for index, record in enumerate(eligible_labels, start=1):
        question = record["question"]
        relevant_ids = set(record["_normalized_chunk_ids"])

        if index % 10 == 0 or index == 1:
            logger.info("Evaluating QA pair %d/%d", index, len(eligible_labels))

        active_strategies = {
            "Dense (Chroma)": dense_search,
            "Sparse (BM25)": sparse_search,
            "Hybrid + BGE Reranker": hybrid_search,
        }

        for strategy_name, strategy in active_strategies.items():
            if strategy_name == "Hybrid + BGE Reranker":
                # Step 1: Cast a wide net with Hybrid Search (Top-50)
                hybrid_results = hybrid_search.search(
                    query=question,
                    documents=documents,
                    limit=20,
                )
                # Step 2: Surgically filter down to Top-5 using the Reranker
                results = reranker.rerank(
                    query=question,
                    results=hybrid_results,
                    top_k=5,
                )
            else:
                # Standard baseline search limit
                results = strategy.search(
                    query=question,
                    documents=documents,
                    limit=5,
                )

            # Step 3: Extract IDs and calculate metrics
            retrieved_ids = []
            for item in results:
                c_id = item.chunk_id
                if c_id is not None:
                    if isinstance(c_id, list):
                        retrieved_ids.extend([str(x).strip() for x in c_id])
                    else:
                        retrieved_ids.append(str(c_id).strip())

            metrics = evaluate_query(
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=relevant_ids,
                k_values=[1, 3, 5],
            )

            evaluation_results[strategy_name].append({
                "question_number": index,
                "doc_id": record.get("doc_id"),
                "question_id": record.get("question_id"),
                "question": question,
                "relevant_chunk_ids": list(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": metrics,
            })

    # Compute aggregate average metrics per strategy
    averages = {}
    metric_names = [
        "precision@1", "precision@3", "precision@5",
        "recall@1", "recall@3", "recall@5",
        "mrr@1", "mrr@3", "mrr@5",
    ]

    print("\n" + "=" * 80)
    print(f"{'Metric':<20} " + " ".join([f"{name:>18}" for name in strategy_names]))
    print("-" * 80)

    for metric in metric_names:
        averages[metric] = {}
        row_str = f"{metric:<20}"
        for strategy_name in strategy_names:
            records = evaluation_results[strategy_name]
            avg_val = sum(r["metrics"][metric] for r in records) / len(records) if records else 0.0
            averages[metric][strategy_name] = avg_val
            row_str += f"{avg_val:>19.4f}"
        print(row_str)

    print("=" * 80)

    # Save output report
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary_averages": averages,
        "detailed_results": evaluation_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("Evaluation report saved to: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies with BGE Reranker.")
    parser.add_argument("--input", required=True, help="Path to documents JSON file.")
    parser.add_argument("--labels", required=True, help="Path to evaluation labels JSON file.")
    parser.add_argument("--output", required=True, help="Path to save evaluation report JSON.")

    args = parser.parse_args()
    evaluate_retrieval_strategies(args.input, args.labels, args.output)


if __name__ == "__main__":
    main()