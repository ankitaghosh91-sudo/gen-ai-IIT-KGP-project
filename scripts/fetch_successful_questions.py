import argparse
import json
from pathlib import Path
from typing import Any

from scripts.retrieve_analysis_script import _build_rank_map, get_rerank_rank


def fetch_successful_questions(json_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    items = data.get("items", [])

    complete_successes: list[dict[str, Any]] = []
    partial_successes: list[dict[str, Any]] = []

    for item in items:
        eval_id = item.get("eval_id")
        doc_id = item.get("doc_id")
        question_id = item.get("question_id")
        question = item.get("question")
        gold_ids = item.get("gold_chunk_ids", [])

        strat_map = {
            row["chunk_id"]: row
            for row in item.get("table_1_strategies", [])
            if row.get("chunk_id") is not None
        }
        raw_dense_map = _build_rank_map(item.get("raw", {}).get("dense", []))
        raw_sparse_map = _build_rank_map(item.get("raw", {}).get("sparse", []))
        raw_hybrid_map = _build_rank_map(item.get("raw", {}).get("hybrid", []))

        per_gold_status: list[dict[str, Any]] = []
        retrieved_count = 0
        any_reranker_boost = False

        for gid in gold_ids:
            s_row = strat_map.get(gid)
            d_rank = s_row.get("dense_rank") if s_row else raw_dense_map.get(gid)
            sp_rank = s_row.get("sparse_rank") if s_row else raw_sparse_map.get(gid)
            h_rank = s_row.get("hybrid_rank") if s_row else raw_hybrid_map.get(gid)
            r_rank = get_rerank_rank(item, gid)

            in_hybrid_pool = gid in strat_map or gid in raw_hybrid_map
            if in_hybrid_pool or r_rank is not None:
                retrieved_count += 1

            pre_rerank_rank = h_rank
            if pre_rerank_rank is None:
                candidates = [r for r in (d_rank, sp_rank) if r is not None]
                pre_rerank_rank = min(candidates) if candidates else None

            rank_gain = (
                (pre_rerank_rank - r_rank)
                if pre_rerank_rank is not None and r_rank is not None
                else None
            )
            if rank_gain is not None and rank_gain > 0:
                any_reranker_boost = True

            per_gold_status.append(
                {
                    "gold_chunk_id": gid,
                    "retrieved": bool(in_hybrid_pool or r_rank is not None),
                    "dense_rank": d_rank,
                    "sparse_rank": sp_rank,
                    "hybrid_rank": h_rank,
                    "rerank_rank": r_rank,
                    "pre_rerank_rank": pre_rerank_rank,
                    "rank_gain": rank_gain,
                }
            )

        if not gold_ids:
            continue

        entry = {
            "item": item.get("item"),
            "eval_id": eval_id,
            "doc_id": doc_id,
            "question_id": question_id,
            "question": question,
            "retrieved_gold_count": retrieved_count,
            "total_gold_count": len(gold_ids),
            "has_reranker_boost": any_reranker_boost,
            "gold_chunk_ids": gold_ids,
            "gold_details": per_gold_status,
        }

        if retrieved_count == len(gold_ids):
            entry["status"] = "complete_success"
            complete_successes.append(entry)
        elif 0 < retrieved_count < len(gold_ids):
            entry["status"] = "partial_success"
            partial_successes.append(entry)

    return {
        "total_questions": len(items),
        "complete_success_count": len(complete_successes),
        "partial_success_count": len(partial_successes),
        "combined_success_count": len(complete_successes) + len(partial_successes),
        "complete_successes": complete_successes,
        "partial_successes": partial_successes,
    }


def format_success_report(report: dict[str, Any]) -> str:
    lines = [
        f"Total Questions Evaluated : {report['total_questions']}",
        f"Complete Successes (Full) : {report['complete_success_count']}",
        f"Partial Successes         : {report['partial_success_count']}",
        f"Combined Non-Zero Hits    : {report['combined_success_count']}",
        "",
        "=== COMPLETE SUCCESSES (All Gold Chunks Retrieved) ===",
    ]
    for cs in report["complete_successes"]:
        boost_tag = " [RERANKER BOOSTED]" if cs["has_reranker_boost"] else ""
        lines.append(
            f"- [{cs['eval_id']}] doc={cs['doc_id']} qid={cs['question_id']} "
            f"({cs['retrieved_gold_count']}/{cs['total_gold_count']}){boost_tag}"
        )
        lines.append(f"  Q: {cs['question']}")

    lines.append("\n=== PARTIAL SUCCESSES (Some Gold Chunks Retrieved) ===")
    for ps in report["partial_successes"]:
        boost_tag = " [RERANKER BOOSTED]" if ps["has_reranker_boost"] else ""
        lines.append(
            f"- [{ps['eval_id']}] doc={ps['doc_id']} qid={ps['question_id']} "
            f"({ps['retrieved_gold_count']}/{ps['total_gold_count']}){boost_tag}"
        )
        lines.append(f"  Q: {ps['question']}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract complete and partial retrieval success questions from trace JSON."
    )
    parser.add_argument(
        "--input",
        default="data/evaluation/500_docs_parent_child/reports/retrieval_reranker_trace_200.json",
        help="Path to input retrieval/reranker trace JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="data/evaluation/500_docs_parent_child/reports/successful_questions_200.json",
        help="Path to write filtered success JSON.",
    )
    parser.add_argument(
        "--text-output",
        default=None,
        help="Optional path to write plain-text summary report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = fetch_successful_questions(args.input)
    summary_text = format_success_report(report)

    print(summary_text, end="")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.text_output:
        out_txt = Path(args.text_output)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(summary_text, encoding="utf-8")

    print(f"Saved JSON report to {out_json}")
    if args.text_output:
        print(f"Saved text summary to {out_txt}")


if __name__ == "__main__":
    main()