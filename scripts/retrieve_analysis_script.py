import argparse
import json
from pathlib import Path
from typing import Any


def get_rerank_rank(item: dict[str, Any], chunk_id: str) -> int | None:
    """Return reranker rank from raw.reranker_output (up to top_k=40), falling back to table_2."""
    for entry in item.get("raw", {}).get("reranker_output", []):
        if entry.get("chunk_id") == chunk_id:
            return entry.get("rank")
    for entry in item.get("table_2_reranker", []):
        if entry.get("chunk_id") == chunk_id:
            return entry.get("rerank_rank")
    return None


def _build_rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    for row in rows:
        cid = row.get("chunk_id")
        rank = row.get("rank")
        if cid is not None and rank is not None:
            rank_map[cid] = rank
    return rank_map


def analyze_trace(json_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    items = data.get("items", [])

    complete_failures: list[dict[str, Any]] = []
    partial_failures: list[dict[str, Any]] = []
    reranker_improvements: list[dict[str, Any]] = []
    reranker_regressions: list[dict[str, Any]] = []

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

        per_gold_status = []
        retrieved_count = 0

        for gid in gold_ids:
            s_row = strat_map.get(gid)
            d_rank = s_row.get("dense_rank") if s_row else raw_dense_map.get(gid)
            sp_rank = s_row.get("sparse_rank") if s_row else raw_sparse_map.get(gid)
            h_rank = s_row.get("hybrid_rank") if s_row else raw_hybrid_map.get(gid)
            r_rank = get_rerank_rank(item, gid)

            in_hybrid_pool = gid in strat_map or gid in raw_hybrid_map
            if in_hybrid_pool or r_rank is not None:
                retrieved_count += 1

            # Baseline pre-rerank rank: prefer hybrid_rank, fallback to best available dense/sparse
            pre_rerank_rank = h_rank
            if pre_rerank_rank is None:
                candidates = [r for r in (d_rank, sp_rank) if r is not None]
                pre_rerank_rank = min(candidates) if candidates else None

            if pre_rerank_rank is not None and r_rank is not None:
                # Smaller rank number = higher rank (e.g., 25 -> 4 is an improvement of +21)
                rank_gain = pre_rerank_rank - r_rank
                record = {
                    "item": item.get("item"),
                    "eval_id": eval_id,
                    "doc_id": doc_id,
                    "question_id": question_id,
                    "question": question,
                    "gold_chunk_id": gid,
                    "dense_rank": d_rank,
                    "sparse_rank": sp_rank,
                    "hybrid_rank": h_rank,
                    "rerank_rank": r_rank,
                    "rank_gain": rank_gain,
                }
                if rank_gain > 0:
                    reranker_improvements.append(record)
                elif rank_gain < 0:
                    reranker_regressions.append(record)

            per_gold_status.append({
                "gold_chunk_id": gid,
                "dense_rank": d_rank,
                "sparse_rank": sp_rank,
                "hybrid_rank": h_rank,
                "rerank_rank": r_rank,
            })

        summary_entry = {
            "item": item.get("item"),
            "eval_id": eval_id,
            "doc_id": doc_id,
            "question_id": question_id,
            "question": question,
            "gold_chunk_ids": gold_ids,
            "gold_details": per_gold_status,
        }

        if len(gold_ids) > 0 and retrieved_count == 0:
            complete_failures.append(summary_entry)
        elif 0 < retrieved_count < len(gold_ids):
            partial_failures.append(summary_entry)

    reranker_improvements.sort(key=lambda x: x["rank_gain"], reverse=True)
    reranker_regressions.sort(key=lambda x: x["rank_gain"])

    return {
        "total_questions": len(items),
        "complete_failure_count": len(complete_failures),
        "partial_failure_count": len(partial_failures),
        "reranker_improvement_count": len(reranker_improvements),
        "reranker_regression_count": len(reranker_regressions),
        "complete_failures": complete_failures,
        "partial_failures": partial_failures,
        "reranker_improvements": reranker_improvements,
        "reranker_regressions": reranker_regressions,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Total Questions Evaluated : {report['total_questions']}",
        f"Complete Failures         : {report['complete_failure_count']}",
        f"Partial Failures          : {report['partial_failure_count']}",
        f"Reranker Rank Improvements: {report['reranker_improvement_count']}",
        f"Reranker Rank Regressions : {report['reranker_regression_count']}",
        "",
        "=== COMPLETE FAILURES (No Gold Chunk Retrieved) ===",
    ]
    for cf in report["complete_failures"]:
        lines.append(
            f"- [{cf['eval_id']}] doc={cf['doc_id']} qid={cf['question_id']} | gold={cf['gold_chunk_ids']}"
        )
        lines.append(f"  Q: {cf['question']}")

    lines.append("\n=== RERANKER RANK IMPROVEMENTS (Pre-Rerank Rank -> Rerank Rank) ===")
    for imp in report["reranker_improvements"]:
        lines.append(
            f"- [{imp['eval_id']}] chunk={imp['gold_chunk_id']} | "
            f"dense={imp['dense_rank']} sparse={imp['sparse_rank']} "
            f"hybrid={imp['hybrid_rank']} -> rerank={imp['rerank_rank']} "
            f"(gain: +{imp['rank_gain']})"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze retrieval and reranker trace JSON reports."
    )
    parser.add_argument(
        "--input",
        default="data/evaluation/500_docs_parent_child/reports/retrieval_reranker_trace_200.json",
        help="Path to input retrieval/reranker trace JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="data/evaluation/500_docs_parent_child/reports/trace_analysis_report_200.json",
        help="Path to write structured analysis JSON.",
    )
    parser.add_argument(
        "--text-output",
        default=None,
        help="Optional path to write plain-text summary report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_trace(args.input)
    summary_text = format_report(report)

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
        print(f"Saved text summary to {args.text_output}")


if __name__ == "__main__":
    main()