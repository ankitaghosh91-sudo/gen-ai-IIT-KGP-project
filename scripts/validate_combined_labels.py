
import argparse
import json
from pathlib import Path
from collections import Counter


DEFAULT_INPUT = Path("data/evaluation/combined_relevance_labels.json")


def validate_labels(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Combined labels must contain a JSON list.")

    print("Input file:", input_path)
    print("\nTotal QA pairs:", len(records))

    print("\nQA pairs by label source:")
    print(dict(Counter(r["label_source"] for r in records)))

    fields = [
        "relevant_chunk_ids",
        "partially_relevant_chunk_ids",
        "irrelevant_chunk_ids",
        "uncertain_chunk_ids",
    ]

    print("\nChunk label counts:")
    for field in fields:
        total = sum(len(r[field]) for r in records)
        print(f"Total {field}: {total}")

    print("\nQA pairs without strictly relevant chunks:")
    no_relevant = 0

    for record in records:
        if not record["relevant_chunk_ids"]:
            no_relevant += 1
            print(
                record["doc_id"],
                record["question_id"],
                "Source:", record["label_source"],
                "Partial:", record["partially_relevant_chunk_ids"],
                "Uncertain:", record["uncertain_chunk_ids"],
            )

    print("Total QA pairs without relevant chunks:", no_relevant)

    print("\nDuplicate relevant chunk IDs:")
    duplicates = 0

    for record in records:
        chunks = record["relevant_chunk_ids"]

        if len(chunks) != len(set(chunks)):
            duplicates += 1
            print(record["doc_id"], record["question_id"], chunks)

    print("QA pairs with duplicate relevant chunks:", duplicates)


def main():
    parser = argparse.ArgumentParser(
        description="Validate combined relevance labels."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to combined relevance labels.",
    )

    args = parser.parse_args()
    validate_labels(args.input)


if __name__ == "__main__":
    main()
