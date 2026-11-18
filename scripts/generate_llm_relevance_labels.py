
"""Generate or resume LLM-based relevance labels for semantic candidates."""

import argparse
import json
from pathlib import Path

from src.evaluation.relevance_judge import RelevanceJudge


DEFAULT_INPUT = Path("data/evaluation/semantic_candidates.json")
DEFAULT_OUTPUT = Path("data/evaluation/llm_relevance_labels.json")


def generate_llm_labels(input_path, output_path):

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Semantic candidates not found: {input_path}"
        )

    records = json.loads(input_path.read_text(encoding="utf-8"))

    # Calculate the total number of candidate judgments.
    total_candidates = sum(
        len(record["candidates"]) for record in records
    )

    # Load previous results if they exist.
    if output_path.exists():
        results = json.loads(
            output_path.read_text(encoding="utf-8")
        )
    else:
        results = []

    # Map existing results to their unique candidate identity.
    result_map = {
        (
            item["doc_id"],
            item["question_id"],
            item["chunk_id"],
        ): item
        for item in results
    }

    judge = RelevanceJudge()

    processed = 0
    skipped = 0
    successful = 0
    failed = 0

    print(f"QA pairs to process: {len(records)}", flush=True)
    print(f"Total candidate judgments: {total_candidates}", flush=True)
    print(f"Existing results loaded: {len(results)}", flush=True)

    for record in records:

        for rank, candidate in enumerate(record["candidates"], start=1):

            processed += 1

            key = (
                record["doc_id"],
                record["question_id"],
                candidate["chunk_id"],
            )

            result_record = result_map.get(key)

            # Skip candidates that already have a successful judgment.
            if (
                result_record
                and result_record.get("llm_label") in judge.VALID_LABELS
                and not result_record.get("error")
            ):
                skipped += 1

                if processed % 100 == 0:
                    print(
                        f"Progress: {processed}/{total_candidates} | "
                        f"Successful: {successful} | "
                        f"Skipped: {skipped} | "
                        f"Failed: {failed}",
                        flush=True,
                    )

                continue

            # Create a new record if this candidate has never been processed.
            if result_record is None:
                result_record = {
                    "doc_id": record["doc_id"],
                    "question_id": record["question_id"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "chunk_id": candidate["chunk_id"],
                    "rank": rank,
                    "similarity_score": candidate["similarity_score"],
                    "llm_label": None,
                    "reason": None,
                    "error": None,
                }

                results.append(result_record)
                result_map[key] = result_record

            print(
                f"\nProcessing candidate {processed}/{total_candidates}",
                flush=True,
            )
            print(
                f"Document: {record['doc_id']} | "
                f"Question: {record['question_id']} | "
                f"Rank: {rank}",
                flush=True,
            )

            try:
                judgment = judge.judge(
                    question=record["question"],
                    expected_answer=record["gold_answer"],
                    candidate_chunk=candidate["chunk_text"],
                )

                result_record["llm_label"] = judgment["label"]
                result_record["reason"] = judgment["reason"]
                result_record["error"] = None

                successful += 1

                print(
                    f"Result: {judgment['label']} | "
                    f"Similarity: {candidate['similarity_score']:.4f}",
                    flush=True,
                )

            except Exception as exc:

                result_record["llm_label"] = None
                result_record["reason"] = None
                result_record["error"] = (
                    f"{type(exc).__name__}: {str(exc)}"
                )

                failed += 1

                print(f"ERROR: {exc}", flush=True)

            # Persist progress after every attempted candidate.
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if processed % 100 == 0:
                print(
                    f"Progress: {processed}/{total_candidates} | "
                    f"Successful: {successful} | "
                    f"Skipped: {skipped} | "
                    f"Failed: {failed}",
                    flush=True,
                )

    print("\n========== SUMMARY ==========")
    print("Total candidates:", total_candidates)
    print("Already successful and skipped:", skipped)
    print("Judgments successful this run:", successful)
    print("Judgment failures this run:", failed)
    print("Results saved to:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate or resume LLM relevance labels."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to semantic candidates JSON",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save LLM relevance labels",
    )

    args = parser.parse_args()

    generate_llm_labels(
        input_path=args.input,
        output_path=args.output,
    )
