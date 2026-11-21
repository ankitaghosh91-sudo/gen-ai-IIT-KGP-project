import json
from pathlib import Path

INPUT_FILE = Path("data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json")
FAILURE_REPORT_FILE = Path("data/evaluation/500_docs_parent_child/failure_analysis_report.json")
BUCKET_NAME = "all_strategies_none"
SAMPLE_LIMIT: int | None = 10  # Set to None to inspect all 65 records


def get_question_number(item: dict, fallback: int) -> int:
    q_num = item.get("question_number")
    if isinstance(q_num, int) and not isinstance(q_num, bool):
        return q_num
    if isinstance(q_num, str) and q_num.isdigit():
        return int(q_num)
    digits = "".join(ch for ch in str(item.get("question_id", "")) if ch.isdigit())
    return int(digits) if digits else fallback


def build_detail_lookup(detailed: dict) -> dict[str, dict[int, dict]]:
    lookup: dict[str, dict[int, dict]] = {}
    for strat_key, entries in detailed.items():
        q_map: dict[int, dict] = {}
        for idx, item in enumerate(entries, start=1):
            q_num = get_question_number(item, fallback=idx)
            q_map[q_num] = item
        lookup[strat_key] = q_map
    return lookup


def main():
    if not FAILURE_REPORT_FILE.exists():
        raise FileNotFoundError(f"Failure report not found: {FAILURE_REPORT_FILE}")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {INPUT_FILE}")

    report = json.loads(FAILURE_REPORT_FILE.read_text(encoding="utf-8"))
    eval_data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    records = report.get("failures_by_bucket", {}).get(BUCKET_NAME, [])
    detail_lookup = build_detail_lookup(eval_data.get("detailed_results", {}))

    selected = records if SAMPLE_LIMIT is None else records[:SAMPLE_LIMIT]
    print(f"Showing {len(selected)} of {len(records)} records in '{BUCKET_NAME}':\n")

    for rec in selected:
        q_num = rec["question_number"]
        qid = rec.get("question_id", "")
        doc_id = rec.get("doc_id", "unknown")
        question = rec.get("question", "")
        rel_ids = rec.get("relevant_chunk_ids", [])

        print("=" * 80)
        print(f"q{q_num} ({qid}) | doc_id: {doc_id}")
        print(f"Question: {question}")
        print(f"Relevant chunks ({len(rel_ids)}): {rel_ids}")

        for strat_key, q_map in detail_lookup.items():
            item = q_map.get(q_num)
            if not item:
                print(f"  - {strat_key}: MISSING entry")
                continue
            retrieved = item.get("retrieved_chunk_ids", [])
            top3 = retrieved[:3]
            print(f"  - {strat_key:<22}: retrieved={len(retrieved):<3} | top3={top3}")


if __name__ == "__main__":
    main()