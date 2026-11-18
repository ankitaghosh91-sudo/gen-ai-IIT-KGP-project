
"""Download a configurable number of records and images from Hugging Face."""

import argparse
from pathlib import Path

from src.data import load_dataset_subset, save_sample


def main():
    parser = argparse.ArgumentParser(
        description="Download a bounded sample from the finance dataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of dataset records to download (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sample",
        help="Directory in which to save the downloaded dataset.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download again even if the output manifest already exists.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be a positive integer")

    output_dir = Path(args.output_dir)
    manifest = output_dir / "documents.json"

    if manifest.exists() and not args.refresh:
        print(
            f"Dataset already exists at {manifest}. "
            "Use --refresh to download it again."
        )
        return

    print(f"Downloading up to {args.limit} records...")
    print(f"Output directory: {output_dir}")

    rows = load_dataset_subset(limit=args.limit)
    documents = save_sample(rows, str(output_dir))

    unique_doc_ids = len({document.doc_id for document in documents})
    total_qa_pairs = sum(len(document.qa_pairs) for document in documents)

    print("\n========== DOWNLOAD SUMMARY ==========")
    print(f"Records downloaded: {len(documents)}")
    print(f"Unique document IDs: {unique_doc_ids}")
    print(f"Duplicate document IDs: {len(documents) - unique_doc_ids}")
    print(f"Total QA pairs: {total_qa_pairs}")
    print(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    main()
