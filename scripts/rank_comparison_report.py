import json
from pathlib import Path

INPUT_FILE = Path("data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json")
OUTPUT_TXT = Path("data/evaluation/500_docs_parent_child/rank_comparison_report.txt")

# Map strategy keys in the JSON to clean labels
STRATEGY_ORDER = [
    ("Dense (Chroma)", "dense"),
    ("Sparse (BM25)", "sparse"),
    ("Hybrid", "hybrid"),
    ("Hybrid + BGE Reranker", "reranker"),
]


def get_first_relevant_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> str:
    """Return 1-based rank of the highest-ranked relevant chunk, or 'None' if not found."""
    relevant_set = set(relevant_ids)
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return str(idx)
    return "None"


def extract_doc_id(relevant_ids: list[str]) -> str:
    """Extract document ID (e.g., '7767' from '7767.jpeg:child:0:1')."""
    if not relevant_ids:
        return "unknown"
    first_id = relevant_ids[0]
    base = first_id.split(":")[0]  # e.g., '7767.jpeg'
    return base.split(".")[0]        # e.g., '7767'


def get_question_number(item: dict, fallback: int) -> int:
    """Extract numeric question_number safely, falling back to digits in question_id."""
    q_num = item.get("question_number")
    if isinstance(q_num, int) and not isinstance(q_num, bool):
        return q_num
    if isinstance(q_num, str) and q_num.isdigit():
        return int(q_num)

    qid = item.get("question_id", "")
    digits = "".join(ch for ch in str(qid) if ch.isdigit())
    return int(digits) if digits else fallback


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input evaluation file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    detailed = data.get("detailed_results", {})

    # Determine available strategies in preferred order (plus any unexpected keys)
    ordered_keys = [k for k, _ in STRATEGY_ORDER if k in detailed]
    extra_keys = [k for k in detailed if k not in dict(STRATEGY_ORDER)]
    all_keys = ordered_keys + extra_keys

    label_map = dict(STRATEGY_ORDER)
    for k in extra_keys:
        label_map[k] = k.lower().replace(" ", "_")

    # Index results per strategy by numeric question_number
    by_strategy: dict[str, dict[int, dict]] = {}
    all_q_nums: set[int] = set()

    for strat_key in all_keys:
        entries = detailed.get(strat_key, [])
        q_map: dict[int, dict] = {}
        for idx, item in enumerate(entries, start=1):
            q_num = get_question_number(item, fallback=idx)
            q_map[q_num] = item
            all_q_nums.add(q_num)
        by_strategy[strat_key] = q_map

    lines: list[str] = []

    for q_num in sorted(all_q_nums):
        sample_item = next(
            by_strategy[s][q_num] for s in all_keys if q_num in by_strategy[s]
        )
        relevant_ids = sample_item.get("relevant_chunk_ids", [])
        qid = sample_item.get("question_id", "")
        doc_id = extract_doc_id(relevant_ids)

        rank_parts = []
        for strat_key in all_keys:
            label = label_map[strat_key]
            item = by_strategy[strat_key].get(q_num)
            if item is None:
                rank_str = "MISSING"
            else:
                rank_str = get_first_relevant_rank(
                    item.get("relevant_chunk_ids", relevant_ids),
                    item.get("retrieved_chunk_ids", []),
                )
            rank_parts.append(f"{label} rank {rank_str}")

        qid_part = f" ({qid})" if qid else ""
        line = f"doc id {doc_id:<6} q{q_num:<4}{qid_part} " + ", ".join(rank_parts)
        lines.append(line)
        print(line)

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"\nSaved {len(lines)} rows to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()