import json
from pathlib import Path

INPUT_FILE = Path("data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json")
OUTPUT_JSON = Path("data/evaluation/500_docs_parent_child/failure_analysis_report.json")

STRATEGY_ORDER = [
    ("Dense (Chroma)", "dense"),
    ("Sparse (BM25)", "sparse"),
    ("Hybrid", "hybrid"),
    ("Hybrid + BGE Reranker", "reranker"),
]


def get_first_relevant_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> int | None:
    """Return 1-based rank of highest-ranked relevant chunk, or None if not found."""
    relevant_set = set(relevant_ids)
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return idx
    return None


def extract_doc_id(relevant_ids: list[str]) -> str:
    if not relevant_ids:
        return "unknown"
    return relevant_ids[0].split(":")[0].split(".")[0]


def get_question_number(item: dict, fallback: int) -> int:
    q_num = item.get("question_number")
    if isinstance(q_num, int) and not isinstance(q_num, bool):
        return q_num
    if isinstance(q_num, str) and q_num.isdigit():
        return int(q_num)
    digits = "".join(ch for ch in str(item.get("question_id", "")) if ch.isdigit())
    return int(digits) if digits else fallback


def classify_failure(ranks: dict[str, int | None]) -> str:
    d = ranks.get("dense")
    s = ranks.get("sparse")
    h = ranks.get("hybrid")
    r = ranks.get("reranker")

    if d is None and s is None and h is None and r is None:
        return "all_strategies_none"
    if (d is not None or s is not None or h is not None) and r is None:
        return "found_earlier_lost_at_reranker"
    if (d is not None or s is not None) and h is None:
        return "found_in_base_lost_in_hybrid"
    if r is None:
        return "reranker_none"
    return "partial_or_other_miss"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    detailed = data.get("detailed_results", {})
    active_strategies = [(k, label) for k, label in STRATEGY_ORDER if k in detailed]

    by_label_q: dict[str, dict[int, dict]] = {}
    all_q_nums: set[int] = set()

    for strat_key, label in active_strategies:
        q_map: dict[int, dict] = {}
        for idx, item in enumerate(detailed[strat_key], start=1):
            q_num = get_question_number(item, fallback=idx)
            q_map[q_num] = item
            all_q_nums.add(q_num)
        by_label_q[label] = q_map

    failures_by_bucket: dict[str, list[dict]] = {
        "all_strategies_none": [],
        "found_in_base_lost_in_hybrid": [],
        "found_earlier_lost_at_reranker": [],
        "reranker_none": [],
        "partial_or_other_miss": [],
    }

    total_questions = len(all_q_nums)
    any_miss_count = 0

    for q_num in sorted(all_q_nums):
        sample = next(
            by_label_q[label][q_num]
            for _, label in active_strategies
            if q_num in by_label_q[label]
        )
        relevant_ids = sample.get("relevant_chunk_ids", [])
        doc_id = extract_doc_id(relevant_ids)
        qid = sample.get("question_id", "")

        ranks: dict[str, int | None] = {}
        question_text = sample.get("question", "")

        for _, label in active_strategies:
            item = by_label_q[label].get(q_num)
            if not question_text and item:
                question_text = item.get("question", "")
            if item is None:
                ranks[label] = None
            else:
                ranks[label] = get_first_relevant_rank(
                    item.get("relevant_chunk_ids", relevant_ids),
                    item.get("retrieved_chunk_ids", []),
                )

        # Consider a record "failed" if at least one strategy missed (None)
        if any(v is None for v in ranks.values()):
            any_miss_count += 1
            bucket = classify_failure(ranks)
            record = {
                "question_number": q_num,
                "question_id": qid,
                "doc_id": doc_id,
                "question": question_text,
                "ranks": ranks,
                "relevant_chunk_ids": relevant_ids,
            }
            failures_by_bucket[bucket].append(record)

    print(f"Total questions evaluated: {total_questions}")
    print(f"Questions with >=1 strategy returning None: {any_miss_count}\n")
    print("Failure breakdown:")
    for bucket, items in failures_by_bucket.items():
        print(f"  - {bucket:<34}: {len(items)}")

    report = {
        "total_questions": total_questions,
        "any_strategy_miss_count": any_miss_count,
        "counts_by_bucket": {k: len(v) for k, v in failures_by_bucket.items()},
        "failures_by_bucket": failures_by_bucket,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved failure analysis report to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()