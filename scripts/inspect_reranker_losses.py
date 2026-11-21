import json
from pathlib import Path

INPUT_FILE = Path("data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json")
FAILURE_REPORT_FILE = Path("data/evaluation/500_docs_parent_child/failure_analysis_report.json")
BUCKET_NAME = "found_earlier_lost_at_reranker"


def get_first_relevant_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> int | None:
    relevant_set = set(relevant_ids)
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return idx
    return None


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

    print(f"Inspecting {len(records)} records in '{BUCKET_NAME}':\n")

    for rec in records:
        q_num = rec["question_number"]
        qid = rec.get("question_id", "")
        doc_id = rec.get("doc_id", "unknown")
        question = rec.get("question", "")
        ranks = rec.get("ranks", {})
        rel_ids = rec.get("relevant_chunk_ids", [])

        print("=" * 80)
        print(f"q{q_num} ({qid}) | doc_id: {doc_id} | ranks: {ranks}")
        print(f"Question: {question}")
        print(f"Relevant chunks ({len(rel_ids)}): {rel_ids}")

        # Check whether relevant chunks appear anywhere inside retrieved_chunk_ids per strategy
        for strat_key, q_map in detail_lookup.items():
            item = q_map.get(q_num)
            if not item:
                print(f"  - {strat_key}: MISSING entry")
                continue
            retrieved = item.get("retrieved_chunk_ids", [])
            rel_set = set(rel_ids or item.get("relevant_chunk_ids", []))
            matched_positions = [
                pos for pos, cid in enumerate(retrieved, start=1) if cid in rel_set
            ]
            print(
                f"  - {strat_key:<22}: retrieved={len(retrieved):<3} | "
                f"first_rank={get_first_relevant_rank(list(rel_set), retrieved)} | "
                f"all_hit_ranks={matched_positions}"
            )


if __name__ == "__main__":
    main()