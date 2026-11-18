
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.config import AppConfig
from src.data.loader import load_saved_sample
from src.pipeline.factory import _build_chunker
from src.strategies.search import build_chunks


LABEL_CATEGORIES = [
    "relevant_chunk_ids",
    "partially_relevant_chunk_ids",
    "irrelevant_chunk_ids",
    "uncertain_chunk_ids",
]


def check_coverage(input_path: str, labels_path: str):
    config = AppConfig.from_env()

    documents = load_saved_sample(path=input_path)
    chunker = _build_chunker(config)

    chunks = build_chunks(documents, chunker)
    indexed_chunk_ids = {chunk.chunk_id for chunk in chunks}

    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    category_ids = defaultdict(set)

    for record in labels:
        for category in LABEL_CATEGORIES:
            category_ids[category].update(record.get(category, []))

    print("Documents:", len(documents))
    print("Available chunks:", len(indexed_chunk_ids))
    print("Label records:", len(labels))

    all_label_ids = set()

    for category in LABEL_CATEGORIES:
        known_ids = category_ids[category]
        available = known_ids & indexed_chunk_ids
        missing = known_ids - indexed_chunk_ids

        all_label_ids.update(known_ids)

        print(f"\n{category}:")
        print("  Unique labeled chunks:", len(known_ids))
        print("  Available chunks:", len(available))
        print("  Missing chunks:", len(missing))

        if missing:
            print("  Missing chunk IDs:")
            for chunk_id in sorted(missing):
                print("   ", chunk_id)

    all_missing = all_label_ids - indexed_chunk_ids

    print("\nOverall:")
    print("  Unique labeled chunk IDs:", len(all_label_ids))
    print("  Unique missing chunk IDs:", len(all_missing))

    return not all_missing


def main():
    parser = argparse.ArgumentParser(
        description="Check evaluation label chunk IDs against the generated corpus."
    )
    parser.add_argument(
        "--input",
        default="data/sample/documents.json",
        help="Path to the saved document sample.",
    )
    parser.add_argument(
        "--labels",
        default="data/evaluation/combined_relevance_labels.json",
        help="Path to the combined relevance labels.",
    )

    args = parser.parse_args()

    check_coverage(args.input, args.labels)


if __name__ == "__main__":
    main()
