from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.pipeline.factory import _build_search


from src.models.records import Document
from src.strategies.search import ChromaSearch, BM25Search, HybridSearch
from src.strategies.reranker import BGEReranker,SearchRerankerPipeline
from src.evaluation.metrics import evaluate_query
import argparse

from dotenv import load_dotenv

load_dotenv()  # loads .env from current/parent working directory into os.environ

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Toggle this to None (or 200) once the 1-record debug run looks right
DEBUG_LIMIT = None 

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

def _call_search(strategy: Any, query: str, documents: list[Any], limit: int = 5) -> list[Any]:
    """Invoke strategy.search uniformly across different strategy signatures."""
    try:
        return strategy.search(query=query, documents=documents, limit=limit)
    except TypeError:
        pass

    try:
        return strategy.search(query=query, documents=documents, top_k=limit)
    except TypeError:
        pass

    try:
        return strategy.search(query, top_k=limit)
    except TypeError:
        pass

    return strategy.search(query)


def _extract_chunk_ids(results: list[Any]) -> list[str]:
    """Normalize chunk IDs from retrieved items (objects or dicts)."""
    retrieved_ids: list[str] = []
    for item in results:
        if isinstance(item, dict):
            raw_id = item.get("chunk_id")
        else:
            raw_id = getattr(item, "chunk_id", None)

        if raw_id is None:
            continue

        if isinstance(raw_id, (list, tuple, set)):
            for cid in raw_id:
                if cid is not None and str(cid).strip():
                    retrieved_ids.append(str(cid).strip())
        else:
            cleaned = str(raw_id).strip()
            if cleaned:
                retrieved_ids.append(cleaned)
    return retrieved_ids

def evaluate_retrieval_strategies(
    docs_path: str,
    labels_path: str,
    output_path: str,
) -> dict[str, Any] | None:
    logger.info("Loading documents from %s", docs_path)
    documents = load_documents(docs_path)

    logger.info("Loading labels from %s", labels_path)
    labels = load_labels(labels_path)

    eligible_labels: list[dict[str, Any]] = []
    for item in labels:
        question = item.get("question")
        chunk_ids = (
            item.get("relevant_chunk_id")
            or item.get("chunk_ids")
            or item.get("relevant_chunk_ids")
        )
        if question and chunk_ids:
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            normalized = [str(c).strip() for c in chunk_ids if str(c).strip()]
            if normalized:
                record_copy = dict(item)
                record_copy["_normalized_chunk_ids"] = normalized
                eligible_labels.append(record_copy)

    if not eligible_labels:
        logger.error("No eligible QA pairs found! Check your label key matching.")
        return None

    if DEBUG_LIMIT is not None:
        eligible_labels = eligible_labels[:DEBUG_LIMIT]
        logger.info("Running in DEBUG mode with %d record(s).", len(eligible_labels))
    else:
        logger.info("Evaluating %d eligible QA pairs.", len(eligible_labels))

    # Build strategies once outside the query loop
    base_config = AppConfig().from_env()
    active_strategies = {
        "Dense (Chroma)": _build_search(replace(base_config, search_strategy="chroma")),
        "Sparse (BM25)": _build_search(replace(base_config, search_strategy="bm25")),
        "Hybrid": _build_search(replace(base_config, search_strategy="hybrid")),
        "Hybrid + BGE Reranker": _build_search(
            replace(base_config, search_strategy="hybrid_reranker")
        ),
    }
    strategy_names = list(active_strategies.keys())
    evaluation_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in strategy_names
    }

    for index, record in enumerate(eligible_labels, start=1):
        question = record["question"]
        relevant_ids = set(record["_normalized_chunk_ids"])

        if index % 10 == 0 or index == 1:
            logger.info("Evaluating QA pair %d/%d", index, len(eligible_labels))

        for strategy_name, strategy in active_strategies.items():
            results = _call_search(strategy=strategy, query=question, documents=documents, limit=base_config.retrieval_limit)
            retrieved_ids = _extract_chunk_ids(results)

            if DEBUG_LIMIT is not None:
                logger.info(
                    "DEBUG | strategy=%s | relevant=%s | retrieved=%s",
                    strategy_name,
                    sorted(relevant_ids),
                    retrieved_ids,
                )

            metrics = evaluate_query(
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=relevant_ids,
                k_values=[1, 3, 5],
            )

            evaluation_results[strategy_name].append(
                {
                    "question_number": index,
                    "doc_id": record.get("doc_id"),
                    "question_id": record.get("question_id"),
                    "question": question,
                    "relevant_chunk_ids": sorted(relevant_ids),
                    "retrieved_chunk_ids": retrieved_ids,
                    "metrics": metrics,
                }
            )

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
    averages: dict[str, dict[str, float]] = {}

    print("\n" + "=" * 95)
    print(f"{'Metric':<20} " + " ".join(f"{name:>18}" for name in strategy_names))
    print("-" * 95)
    for metric in metric_names:
        averages[metric] = {}
        row = f"{metric:<20}"
        for name in strategy_names:
            records = evaluation_results[name]
            avg_val = (
                sum(r["metrics"].get(metric, 0.0) for r in records) / len(records)
                if records
                else 0.0
            )
            averages[metric][name] = avg_val
            row += f"{avg_val:>18.4f}"
        print(row)
    print("=" * 95)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "debug_limit": DEBUG_LIMIT,
        "summary_averages": averages,
        "detailed_results": evaluation_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("Evaluation report saved to: %s", out_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies with BGE Reranker.")
    parser.add_argument("--input", required=True, help="Path to documents JSON file.")
    parser.add_argument("--labels", required=True, help="Path to evaluation labels JSON file.")
    parser.add_argument("--output", required=True, help="Path to save evaluation report JSON.")

    args = parser.parse_args()
    evaluate_retrieval_strategies(args.input, args.labels, args.output)


if __name__ == "__main__":
    main()