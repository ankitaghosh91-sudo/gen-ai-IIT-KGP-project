
"""Find semantic candidates without a successful LLM relevance judgment."""

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "data/evaluation/500_docs/semantic_candidates.json"
)
DEFAULT_LABELS = Path(
    "data/evaluation/500_docs/llm_relevance_labels.json"
)
DEFAULT_OUTPUT = Path(
    "data/evaluation/500_docs/unlabeled_semantic_candidates.json"
)

VALID_LABELS = {
    "relevant",
    "partially_relevant",
    "irrelevant",
    "uncertain",
}


def find_unlabeled_candidates(input_path, labels_path, output_path):

    input_path = Path(input_path)
    labels_path = Path(labels_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Semantic candidates file not found: {input_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"LLM relevance labels file not found: {labels_path}"
        )

    semantic_records = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    label_records = json.loads(
        labels_path.read_text(encoding="utf-8")
    )

    # Index existing judgments by candidate identity.
    label_map = {}

    for item in label_records:
        key = (
            item["doc_id"],
            item["question_id"],
            item["chunk_id"],
        )
        label_map[key] = item

    pending_records = []

    total_candidates = 0
    successful = 0
    missing_records = 0
    failed_or_unlabeled = 0

    for record in semantic_records:

        pending_candidates = []

        for candidate in record["candidates"]:

            total_candidates += 1

            key = (
                record["doc_id"],
                record["question_id"],
                candidate["chunk_id"],
            )

            judgment = label_map.get(key)

            # Candidate already has a valid successful judgment.
            if (
                judgment
                and judgment.get("llm_label") in VALID_LABELS
                and not judgment.get("error")
            ):
                successful += 1
                continue

            # Candidate has no corresponding judgment record.
            if judgment is None:
                missing_records += 1
            else:
                # Record exists, but judgment failed or has no valid label.
                failed_or_unlabeled += 1

            pending_candidates.append(candidate)

        # Keep only QA records containing unfinished candidates.
        if pending_candidates:
            pending_record = {
                key: value
                for key, value in record.items()
                if key != "candidates"
            }

            pending_record["candidates"] = pending_candidates
            pending_records.append(pending_record)

    pending_count = sum(
        len(record["candidates"])
        for record in pending_records
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            pending_records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== AUDIT SUMMARY ==========")
    print("Total semantic candidates:", total_candidates)
    print("Successfully judged:", successful)
    print("Missing judgment records:", missing_records)
    print("Failed or unlabeled records:", failed_or_unlabeled)
    print("Total candidates needing retry:", pending_count)
    print("QA pairs containing pending candidates:", len(pending_records))
    print("Retry file saved to:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Find candidates without successful LLM relevance labels."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to semantic candidates JSON",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to existing LLM relevance labels JSON",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save candidates needing retry",
    )

    args = parser.parse_args()

    find_unlabeled_candidates(
        input_path=args.input,
        labels_path=args.labels,
        output_path=args.output,
    )
