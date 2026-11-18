"""Inspect unmatched relevance labels for one document."""

import json
from pathlib import Path

from src.data import load_saved_sample
from src.strategies.chunking import RecursiveChunking
from src.strategies.search import build_chunks


def inspect_unmatched_labels(doc_id: str):
    documents = load_saved_sample()

    document = next(
        (doc for doc in documents if doc.doc_id == doc_id),
        None,
    )

    if document is None:
        raise ValueError(f"Document not found: {doc_id}")

    labels_path = Path("data/evaluation/relevance_labels.json")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    chunker = RecursiveChunking(chunk_size=500, overlap=100)
    chunks = build_chunks([document], chunker)

    unmatched = [
        item
        for item in labels
        if item["doc_id"] == doc_id
        and not item["relevant_chunks"]
    ]

    print(f"\nDocument: {doc_id}")
    print(f"Unmatched QA pairs: {len(unmatched)}")
    print(f"Available chunks: {len(chunks)}")

    for item in unmatched:
        print("\n" + "=" * 80)
        print("Question ID:", item["question_id"])
        print("Question:", item["question"])
        print("Gold answer:", item["gold_answer"])

        print("\nCandidate chunks:")

        for chunk in chunks:
            print(f"\n[{chunk.chunk_id}]")
            print(chunk.text)


if __name__ == "__main__":
    inspect_unmatched_labels("175.jpeg")