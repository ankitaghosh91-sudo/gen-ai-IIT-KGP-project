
"""Generate initial answer-matching relevance labels for RAG evaluation."""

import argparse
import json
import re
from pathlib import Path

from src.data import load_saved_sample
from src.config import AppConfig
from src.pipeline.factory import _build_chunker
from src.strategies.search import build_chunks


DEFAULT_INPUT = "data/sample/documents.json"
DEFAULT_OUTPUT = "data/evaluation/relevance_labels.json"


def normalize_answer(text: str) -> str:
    """Normalize text for basic answer matching."""

    text = text.lower()
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_labels(input_path: str, output_path: str):

    # 1. Load saved documents and their original QA pairs.
    documents = load_saved_sample(input_path)

    if not documents:
        raise ValueError(
            f"No saved documents found at: {input_path}"
        )

    # 2. Use the same chunking configuration as the RAG pipeline.
    config = AppConfig.from_env()
    chunker = _build_chunker(config)

    evaluation_data = []

    # 3. Process each document independently.
    for doc_index, document in enumerate(documents, start=1):

        chunks = build_chunks([document], chunker)
        

        for qa_index, qa in enumerate(document.qa_pairs):

            normalized_answer = normalize_answer(qa.answer)

            relevant_chunks = []

            # 4. Compare this QA answer against its document's chunks.
            for chunk in chunks:
                print("Chunk Parent Id",chunk.parent_id)
                normalized_chunk = normalize_answer(chunk.text)

                if (
                    normalized_answer
                    and normalized_answer in normalized_chunk
                ):
                    relevant_chunks.append(chunk.chunk_id)

            evaluation_data.append({
                "doc_id": document.doc_id,
                "question_id": f"q{qa_index + 1}",
                "question": qa.question,
                "gold_answer": qa.answer,
                "relevant_chunks": relevant_chunks,
                "label_method": "normalized_exact_match",
            })

        if doc_index % 50 == 0 or doc_index == len(documents):
            print(
                f"Processed {doc_index}/{len(documents)} documents"
            )

    # 5. Save labels for reuse across retrieval experiments.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(evaluation_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matched = sum(
        bool(item["relevant_chunks"])
        for item in evaluation_data
    )

    print("\n========== LABEL GENERATION SUMMARY ==========")
    print("Documents processed:", len(documents))
    print("QA pairs processed:", len(evaluation_data))
    print("QA pairs with matching chunks:", matched)
    print(
        "QA pairs without matching chunks:",
        len(evaluation_data) - matched,
    )
    print("Labels saved to:", output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate initial answer-matching relevance labels."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the saved documents.json manifest.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path where relevance labels will be saved.",
    )

    args = parser.parse_args()

    generate_labels(
        input_path=args.input,
        output_path=args.output,
    )
